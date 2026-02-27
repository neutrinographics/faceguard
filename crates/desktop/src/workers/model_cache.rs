use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use faceguard_core::detection::infrastructure::model_resolver;
use faceguard_core::detection::infrastructure::onnx_yolo_detector;
use faceguard_core::shared::constants::{
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_URL, WHISPER_MODEL_NAME, WHISPER_MODEL_URL,
    YOLO_MODEL_NAME, YOLO_MODEL_URL,
};

/// Shared model cache that resolves models and pre-builds ONNX sessions
/// in the background at startup. Workers can grab pre-built sessions or
/// fall back to building from the cached path.
pub struct ModelCache {
    yolo_path: Arc<ModelSlot>,
    embedding_path: Arc<ModelSlot>,
    whisper_path: Arc<ModelSlot>,
    yolo_session: Arc<SessionSlot>,
}

struct ModelSlot {
    result: Mutex<Option<Result<PathBuf, String>>>,
    ready: Condvar,
    progress: Arc<Mutex<(u64, u64)>>,
}

struct SessionSlot {
    session: Mutex<Option<Arc<Mutex<ort::session::Session>>>>,
    input_size: Mutex<u32>,
    ready: Condvar,
    /// Set to true once the first build attempt completes (success or failure).
    built: Mutex<bool>,
}

impl ModelCache {
    /// Create a new `ModelCache` and begin resolving models in the background.
    pub fn new() -> Arc<Self> {
        let cache = Arc::new(Self {
            yolo_path: Arc::new(ModelSlot::new()),
            embedding_path: Arc::new(ModelSlot::new()),
            whisper_path: Arc::new(ModelSlot::new()),
            yolo_session: Arc::new(SessionSlot::new()),
        });

        let yolo_path_slot = cache.yolo_path.clone();
        let embedding_path_slot = cache.embedding_path.clone();
        let whisper_path_slot = cache.whisper_path.clone();
        let session_slot = cache.yolo_session.clone();
        thread::spawn(move || {
            // Resolve YOLO model path (may download)
            yolo_path_slot.resolve(YOLO_MODEL_NAME, YOLO_MODEL_URL);

            // Pre-build the ONNX session from the resolved path
            if let Some(Ok(ref path)) = *yolo_path_slot.result.lock().unwrap() {
                log::info!("Building YOLO session from {}", path.display());
                let start = std::time::Instant::now();
                if let Ok(session) = onnx_yolo_detector::OnnxYoloDetector::build_session(path) {
                    log::info!("YOLO session built in {:?}", start.elapsed());
                    let input_size = onnx_yolo_detector::session_input_size(&session);
                    *session_slot.input_size.lock().unwrap() = input_size;
                    *session_slot.session.lock().unwrap() = Some(Arc::new(Mutex::new(session)));
                } else {
                    log::warn!("Failed to build YOLO session");
                }
            }
            *session_slot.built.lock().unwrap() = true;
            session_slot.ready.notify_all();

            // Resolve embedding model path
            embedding_path_slot.resolve(EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_URL);

            // Resolve whisper model path
            whisper_path_slot.resolve(WHISPER_MODEL_NAME, WHISPER_MODEL_URL);
        });

        cache
    }

    /// Wait for the YOLO model path. Calls `on_progress(downloaded, total)`
    /// while a download is in progress. Returns early if `cancelled` is set.
    pub fn wait_for_yolo(
        &self,
        on_progress: &dyn Fn(u64, u64),
        cancelled: &AtomicBool,
    ) -> Result<PathBuf, String> {
        self.yolo_path.wait(on_progress, cancelled)
    }

    /// Get the shared YOLO ONNX session, waiting for it to be ready.
    /// Returns `None` only if the build failed. The `Arc` can be cloned freely;
    /// every worker shares the same underlying session.
    pub fn get_yolo_session(&self) -> Option<(Arc<Mutex<ort::session::Session>>, u32)> {
        // Wait for the initial build to complete
        let mut built = self.yolo_session.built.lock().unwrap();
        while !*built {
            let (new_guard, _) = self
                .yolo_session
                .ready
                .wait_timeout(built, Duration::from_millis(100))
                .unwrap();
            built = new_guard;
        }
        drop(built);

        let session = self.yolo_session.session.lock().unwrap().clone();
        let input_size = *self.yolo_session.input_size.lock().unwrap();
        session.map(|s| (s, input_size))
    }

    /// Wait for the Whisper model path.
    pub fn wait_for_whisper(
        &self,
        on_progress: &dyn Fn(u64, u64),
        cancelled: &AtomicBool,
    ) -> Result<PathBuf, String> {
        self.whisper_path.wait(on_progress, cancelled)
    }

    /// Wait for the embedding model path.
    pub fn wait_for_embedding(
        &self,
        on_progress: &dyn Fn(u64, u64),
        cancelled: &AtomicBool,
    ) -> Result<PathBuf, String> {
        self.embedding_path.wait(on_progress, cancelled)
    }
}

impl ModelSlot {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            ready: Condvar::new(),
            progress: Arc::new(Mutex::new((0, 0))),
        }
    }

    fn resolve(&self, name: &str, url: &str) {
        let progress_mutex = self.progress.clone();
        let result = model_resolver::resolve(
            name,
            url,
            None,
            Some(Box::new(move |downloaded, total| {
                *progress_mutex.lock().unwrap() = (downloaded, total);
            })),
        );
        *self.result.lock().unwrap() = Some(result.map_err(|e| e.to_string()));
        self.ready.notify_all();
    }

    fn wait(
        &self,
        on_progress: &dyn Fn(u64, u64),
        cancelled: &AtomicBool,
    ) -> Result<PathBuf, String> {
        let mut guard = self.result.lock().unwrap();
        loop {
            if cancelled.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            if let Some(ref result) = *guard {
                return result.clone();
            }
            // Forward download progress while waiting
            if let Ok(progress) = self.progress.try_lock() {
                let (dl, total) = *progress;
                if total > 0 {
                    on_progress(dl, total);
                }
            }
            let (new_guard, _) = self
                .ready
                .wait_timeout(guard, Duration::from_millis(100))
                .unwrap();
            guard = new_guard;
        }
    }
}

impl SessionSlot {
    fn new() -> Self {
        Self {
            session: Mutex::new(None),
            input_size: Mutex::new(0),
            ready: Condvar::new(),
            built: Mutex::new(false),
        }
    }
}
