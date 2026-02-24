# Performance Audit

## HIGH Impact

### 1. ~~GPU buffers recreated per region per frame~~
**File**: `crates/core/src/blurring/infrastructure/gpu_context.rs:157`

4 GPU buffers + 2 params buffers + 2 bind groups + 3 command encoders created and destroyed per face per frame. With 3 faces at 30fps, that's ~360 buffer creations/second.

**Fix**: Pre-allocate buffers at maximum ROI size and reuse across frames.

---

### 2. ~~GPU sync round-trips per region~~
**File**: `crates/core/src/blurring/infrastructure/gpu_elliptical_blurrer.rs:44`

For each face: extract ROI on CPU, upload to GPU, blur with synchronous readback (`device.poll(Maintain::Wait)`), download, unpack. CPU stalls waiting for GPU; GPU idles during CPU work.

**Fix**: Upload entire frame once, dispatch all region blurs in a single command buffer, download once.

---

### 3. ~~Detection blocks blur on main thread~~
**File**: `crates/core/src/pipeline/blur_faces_use_case.rs:131`

Pipeline is: `reader_thread -> [detect THEN blur on main thread] -> writer_thread`. Detection (~30ms) and blur (~10ms) run sequentially, so the reader starves waiting for the main thread to drain the queue.

**Fix**: Separate detection into its own thread: `reader -> detect_thread -> blur_thread -> writer`.

---

## MEDIUM Impact

### 4. ~~Letterbox tensor allocated per frame~~
**File**: `crates/core/src/detection/infrastructure/onnx_yolo_detector.rs:264`

A 4.7MB `Array4<f32>` tensor is heap-allocated for every detection frame. At 30fps with 50% skip, that's ~70MB/s of allocation churn.

**Fix**: Pre-allocate tensor once in the detector struct and reuse.

---

### 5. ~~YOLO output copied to Vec under lock~~
**File**: `crates/core/src/detection/infrastructure/onnx_yolo_detector.rs:139`

Entire YOLO output tensor (8400 detections x 20 features = ~672KB) is copied to a `Vec` while holding the session mutex.

**Fix**: Process tensor data directly without copying, or copy only after filtering by confidence.

---

### 6. ~~Transposed YOLO creates 8400 tiny Vecs~~
**File**: `crates/core/src/detection/infrastructure/onnx_yolo_detector.rs:164`

For transposed output layout, each of 8400 detections allocates a `Vec<f32>` just to read 5 values.

**Fix**: Use an indexing helper function instead: `get_feat(data, det_idx, feat_idx, num_dets, transposed) -> f32`.

---

### 7. ~~Gaussian blur uses f64~~
**File**: `crates/core/src/blurring/infrastructure/gaussian.rs:25`

Entire separable Gaussian blur operates on f64. For 8-bit pixel processing, f32 is sufficient and ~2x faster due to cache efficiency and SIMD vectorization.

**Fix**: Switch to f32 for the temp buffer and kernel.

---

### 8. ~~Ellipse mask does per-pixel division~~
**File**: `crates/core/src/blurring/infrastructure/cpu_elliptical_blurrer.rs:83`

Ellipse SDF computes `(dx/semi_a)^2 + (dy/semi_b)^2` with two divisions per pixel. For a 200x200 region, that's 40,000 divisions.

**Fix**: Pre-compute `inv_a_sq = 1.0 / (semi_a * semi_a)` and `inv_b_sq = 1.0 / (semi_b * semi_b)`.

---

### 9. ~~ROI buffer allocated per region~~
**File**: `crates/core/src/blurring/infrastructure/cpu_elliptical_blurrer.rs:60`
**File**: `crates/core/src/blurring/infrastructure/cpu_rectangular_blurrer.rs:60`

~120KB `Vec<u8>` allocated and freed for each face region per frame.

**Fix**: Reuse a single buffer across regions (via `RefCell<Vec<u8>>` on the blurrer struct).

---

### 10. ~~Gaussian temp buffer allocated per call~~
**File**: `crates/core/src/blurring/infrastructure/gaussian.rs:39`

~960KB `Vec<f64>` temp buffer allocated every blur call (per region per frame).

**Fix**: Pre-allocate in a reusable kernel struct.

---

### 11. ~~Audio muxing creates temp file with 4x I/O~~
**File**: `crates/core/src/video/infrastructure/ffmpeg_writer.rs:216`

After encoding video, the muxer re-opens both source and output to copy audio into a temp file, then renames. For a 1GB output, this is ~4GB of disk I/O.

**Fix**: Mux audio during initial encoding by adding the audio stream to the output context upfront.

---

### 12. ~~MPEG4 codec instead of H.264~~
**File**: `crates/core/src/video/infrastructure/ffmpeg_writer.rs:73`

MPEG4 Part 2 produces 30-50% larger files than H.264 at the same quality.

**Fix**: Use H.264 (libx264) with fallback to MPEG4.

---

### 13. GPU shader recomputes exp() per pixel per tap
**File**: `crates/core/src/blurring/infrastructure/shaders/gaussian_blur.wgsl:66`

For kernel_radius=100, each pixel computes 201 `exp()` calls. For a 512x512 region: 52M exp() operations.

**Fix**: Pre-compute kernel weights on CPU and pass as a storage buffer.

---

### 14. ~~Three queue.submit() calls per region~~
**File**: `crates/core/src/blurring/infrastructure/gpu_context.rs:254,265,332`

Horizontal blur, buffer copy, and vertical blur each get their own submit. 3 submits per face = 9 per frame with 3 faces.

**Fix**: Record all operations into a single command encoder and submit once.

---

### 15. Lookahead regions cloned every frame
**File**: `crates/core/src/pipeline/blur_faces_use_case.rs:248`

All lookahead region vectors are cloned on every frame flush.

**Fix**: Change `RegionMerger::merge()` to accept `&[&[Region]]` instead of `&[Vec<Region>]`.

---

### 16. Detection cache cloned in blur_worker
**File**: `crates/desktop/src/workers/blur_worker.rs:92`

Full `HashMap<usize, Vec<Region>>` cloned when passed to `CachedFaceDetector`. Scales with video length.

**Fix**: Use `Arc<HashMap<...>>` for shared ownership.

---

### 17. No ONNX intra-op threading configured
**File**: `crates/core/src/detection/infrastructure/onnx_yolo_detector.rs:91`

Only `with_inter_threads(1)` is set. No `with_intra_threads()`, so CPU fallback may underutilize cores.

**Fix**: Set `with_intra_threads(available_parallelism())`.

---

### 18. Channel capacity may be too small
**File**: `crates/core/src/pipeline/blur_faces_use_case.rs:20`

`CHANNEL_CAPACITY = 4` may cause blocking if detection/blur times vary.

**Fix**: Increase to 8 or make configurable.

---

## LOW Impact

### 19. Gaussian kernel recomputed per call
**File**: `crates/core/src/blurring/infrastructure/gaussian.rs:35`

`gaussian_kernel_1d()` called every blur. Redundant since kernel_size is fixed per job.

**Fix**: Pre-compute once in a kernel struct.

---

### 20. Only CoreML EP configured for ONNX
**File**: `crates/core/src/detection/infrastructure/onnx_yolo_detector.rs:92`

No GPU acceleration on Linux/Windows. Falls back to CPU silently.

**Fix**: Add conditional compilation for CUDA (Linux) and DirectML (Windows).

---

### 21. `gpu_available()` creates full GPU context
**File**: `crates/core/src/blurring/infrastructure/blurrer_factory.rs:46`

Creates Instance, Adapter, Device, Pipeline just to return a bool.

**Fix**: Only check adapter availability.

---

### 22. preview_worker leaks TempDir
**File**: `crates/desktop/src/workers/preview_worker.rs:166`

`std::mem::forget(temp_dir)` prevents cleanup. Files persist until process exit.

**Fix**: Store TempDir in app state for proper lifecycle management.

---

### 23. Embedding grouper missing optimizations
**File**: `crates/core/src/detection/infrastructure/embedding_face_grouper.rs:29`

Session built without Level3 optimization or CoreML EP. Runs on CPU by default.

**Fix**: Match YOLO detector configuration.
