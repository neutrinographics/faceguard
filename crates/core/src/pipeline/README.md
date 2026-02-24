# Pipeline (Application Layer)

Orchestrates domain components into end-to-end use cases. Depends only on domain traits — concrete infrastructure is injected by the composition root (CLI or desktop crate).

## Use Cases

### BlurFacesUseCase
Full video pipeline. Wires together a reader, writer, detector, blurrer, region merger, and pipeline executor. The use case itself owns configuration (lookahead depth, blur/exclude ID sets, progress callback, cancellation flag) and delegates execution to a `PipelineExecutor`.

The executor runs a four-stage pipeline: **read → detect → merge/blur → write**. Frames are buffered to a configurable lookahead depth (default 5) before the oldest frame is flushed. This buffering enables `RegionMerger` to see future detections and smoothly interpolate incoming faces.

Cancellation is cooperative: an `AtomicBool` is checked between frames on each thread. Progress is reported via a callback that returns `false` to request cancellation.

### BlurImageUseCase
Simplified single-image pipeline: read one frame, detect, filter regions by track ID, blur, write. No lookahead, no threading, no merging.

### PreviewFacesUseCase
Scans a video to identify all tracked faces. For each track ID, saves the largest detection (by area) as a 256x256 thumbnail. Returns both the saved thumbnails and a complete detection cache (frame index → regions). The cache can be fed to `CachedFaceDetector` in the subsequent blur pass, ensuring that the track IDs the user selected in the UI match exactly what gets blurred.

## Supporting Types

### PipelineExecutor (trait)
Abstracts how the pipeline stages are scheduled. The only production implementation is `ThreadedPipelineExecutor`, but the trait allows tests to inject a simple sequential executor.

### PipelineConfig
Groups runtime configuration passed to the executor: lookahead depth, blur/exclude ID sets, progress callback, and cancellation flag.

### PipelineLogger (trait)
Cross-cutting observer for pipeline events (progress, stage timing, metrics). `NullPipelineLogger` discards everything (used by GUI and tests). `StdoutPipelineLogger` tracks per-stage averages and emits a summary report.

## Infrastructure

### ThreadedPipelineExecutor
Runs four stages on dedicated threads connected by bounded `crossbeam-channel` queues (capacity 8):

```
reader_thread ──→ detect_thread ──→ main_thread ──→ writer_thread
                                    (buffer/merge/blur)
```

Detection and I/O overlap, improving throughput when detection is the bottleneck. The main thread handles buffering, lookahead merging, and blurring sequentially because blurring mutates the frame in-place.

Region filtering (`Region::filter`) happens in the detect thread immediately after detection, before frames enter the merge buffer. This minimizes the data flowing through downstream stages.
