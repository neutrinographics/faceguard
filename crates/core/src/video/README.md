# Video Feature Slice

Reads and writes video and image files, converting between container/codec formats and the domain's RGB `Frame` representation.

## Domain

### VideoReader (trait)
Opens a source file (video or image), provides metadata, and yields frames as an iterator. The iterator pattern allows streaming processing — frames are decoded one at a time rather than loaded entirely into memory.

### VideoWriter (trait)
Opens an output file, accepts frames sequentially, and finalizes on `close()`. Audio stream copying from the source happens during `close()` so frames can be written incrementally.

### ImageWriter (trait)
Writes a single frame to an image file with optional resize. Used by `PreviewFacesUseCase` for face thumbnails.

## Infrastructure

### FfmpegReader
Wraps `ffmpeg-next` for video decoding. Converts frames from native pixel format (typically YUV420p) to RGB at the I/O boundary. Returns `VideoMetadata` including codec, dimensions, FPS, and frame count.

### FfmpegWriter
Wraps `ffmpeg-next` for video encoding. Converts RGB frames back to YUV420p for H.264 encoding. On `close()`, copies the audio stream from the source file (if present) by invoking ffmpeg's muxer.

### ImageFileReader
Reads a single image file as a one-frame video source. Uses `ffmpeg-next` for decoding to maintain consistent RGB conversion across formats.

### ImageFileWriter
Writes frames to image files (JPEG, PNG, etc.) via the `image` crate. Supports optional resize for thumbnail generation.

## Design Decisions

- **RGB internally**: All domain code works with RGB pixel data. YUV/NV12/etc. conversion happens exclusively in the video infrastructure layer. This keeps the domain layer free of codec concerns.
- **Images as single-frame video**: `VideoReader` and `VideoMetadata` are reused for images (`total_frames=1`, `fps=0.0`), avoiding a separate image pipeline. The same use cases handle both.
