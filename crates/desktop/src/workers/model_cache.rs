use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use video_blur_core::detection::infrastructure::model_resolver;
use video_blur_core::shared::constants::{
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_URL, YOLO_MODEL_NAME, YOLO_MODEL_URL,
};

/// Shared model cache that resolves models in the background at startup.
/// Workers can grab pre-resolved paths or wait for in-progress resolution.
pub struct ModelCache {
    yolo: Arc<ModelSlot>,
    embedding: Arc<ModelSlot>,
}

struct ModelSlot {
    result: Mutex<Option<Result<PathBuf, String>>>,
    ready: Condvar,
    progress: Arc<Mutex<(u64, u64)>>,
}

impl ModelCache {
    /// Create a new `ModelCache` and begin resolving models in the background.
    pub fn new() -> Arc<Self> {
        let cache = Arc::new(Self {
            yolo: Arc::new(ModelSlot::new()),
            embedding: Arc::new(ModelSlot::new()),
        });

        let yolo_slot = cache.yolo.clone();
        let embedding_slot = cache.embedding.clone();
        thread::spawn(move || {
            yolo_slot.resolve(YOLO_MODEL_NAME, YOLO_MODEL_URL);
            embedding_slot.resolve(EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_URL);
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
        self.yolo.wait(on_progress, cancelled)
    }

    /// Wait for the embedding model path.
    pub fn wait_for_embedding(
        &self,
        on_progress: &dyn Fn(u64, u64),
        cancelled: &AtomicBool,
    ) -> Result<PathBuf, String> {
        self.embedding.wait(on_progress, cancelled)
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
