# Detection Feature Slice

Detects faces in video frames, assigns persistent track IDs, and converts raw detections into blur-ready regions.

## Domain

### FaceDetector (trait)
Primary interface. Takes a `&Frame`, returns `Vec<Region>`. Stateful (`&mut self`) because implementations maintain cross-frame tracking state (ByteTrack assigns persistent IDs by correlating detections across consecutive frames).

### FaceGrouper (trait)
Groups face crops by identity. Used after a preview pass to cluster track IDs that likely represent the same person, so the user can select "blur person X" rather than individual track segments.

### FaceLandmarks
5-point landmarks (left eye, right eye, nose, left mouth, right mouth) with weighted centroid computation. The nose receives 3x weight to keep the center stable even when mouth landmarks are occluded. Profile ratio (0.0=frontal, 1.0=side) is derived from nose offset relative to eye span.

### FaceRegionBuilder
Converts a bounding box + optional landmarks into a `Region`. Key behaviors:
- **Profile compensation**: As profile ratio increases, region width expands toward height and center blends toward the bounding box center (away from landmarks), preventing partial face exposure on turned heads.
- **Minimum width ratio** (0.8): Narrow detections are widened to at least 80% of height.
- **Padding** (0.4): Applied symmetrically around the computed dimensions.
- **Clamping**: Output is clamped to frame bounds, but unclamped coordinates are preserved for edge-aware ellipse rendering.
- **Temporal smoothing**: Optionally delegates to `RegionSmootherInterface` for EMA-based jitter reduction.

### RegionSmoother
Per-track EMA smoother. First observation passes through unchanged; subsequent observations blend with history: `ema[t] = 0.6 * current + 0.4 * ema[t-1]`. Regions without a `track_id` bypass smoothing entirely.

### RegionMerger
Merges current-frame detections with lookahead frames for smooth face transitions:
1. Current regions are kept as-is.
2. Lookahead regions with new track IDs are interpolated toward the nearest frame edge (creating slide-in animation).
3. Track ID deduplication ensures current frame wins over lookahead.
4. Final IoU deduplication (threshold 0.3) removes remaining overlaps.

The interpolation strength `t = (idx+1) / (total+1)` increases with temporal distance. Only regions whose center is within 25% of a frame edge are interpolated.

## Infrastructure

### OnnxYoloDetector
YOLO11-pose face detector via ONNX Runtime. Pipeline per frame:
1. **Letterbox** — Resize with aspect-ratio padding to model input size (typically 640x640).
2. **Inference** — Run ONNX session, producing bounding boxes + 5-point landmarks + confidences.
3. **NMS** — Non-maximum suppression (IoU threshold 0.45) to remove duplicate detections.
4. **ByteTrack** — Multi-object tracker assigns persistent `track_id`s by correlating detections across frames via IoU matching. Lost tracks survive up to 30 frames (~1s at 30fps).
5. **FaceRegionBuilder** — Converts tracked detections into blur regions with smoothing.

### SkipFrameDetector
Decorator that runs the inner detector every N frames. On skipped frames, regions are linearly extrapolated using per-track velocity computed from the two most recent real detections. Reduces inference cost proportionally while maintaining smooth motion.

### CachedFaceDetector
Replays pre-computed detections by frame index. Used when the preview pass has already detected all faces — guarantees track IDs match exactly what the user saw in the preview UI.

### HistogramFaceGrouper
Groups faces by HSV histogram correlation (Pearson r) with union-find clustering. No model required.

### EmbeddingFaceGrouper
Groups faces using a dedicated ONNX face embedding model (w600k_r50). Cosine similarity between L2-normalized embeddings, clustered via union-find.

### model_resolver
Resolves ONNX model files. Checks user cache directory first, then a bundled path, then downloads from a GitHub release URL. Writes atomically (temp file + rename) to prevent partial downloads.
