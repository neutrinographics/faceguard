# Blurring Feature Slice

Applies Gaussian blur to face regions within video frames.

## Domain

### FrameBlurrer (trait)
Takes `&self` (stateless) and `&mut Frame` + `&[Region]`. Modifies frame pixels in-place within each region. The `&mut Frame` contract avoids allocation — the caller owns the buffer and the blurrer writes directly into it.

## Infrastructure

All implementations use separable Gaussian blur (two 1D passes instead of a 2D convolution) for O(n*k) rather than O(n*k^2) cost per pixel.

### CPU Implementations
- `CpuRectangularBlurrer` — Blurs the rectangular bounding box of each region.
- `CpuEllipticalBlurrer` — Same blur kernel, but masks pixels outside the inscribed ellipse using the region's `ellipse_center_in_roi()` and `ellipse_axes()` for natural-looking oval blur shapes. The ellipse uses unclamped dimensions so it extends off frame edges smoothly.

### GPU Implementations
- `GpuRectangularBlurrer` / `GpuEllipticalBlurrer` — wgpu compute shader implementations. A shared `GpuContext` manages the device, queue, shader module, and pipelines. ROIs are batched into a single GPU dispatch to minimize CPU-GPU round-trips.

### blurrer_factory
Entry point for downstream crates. `create_blurrer(shape, kernel_size)` probes for a wgpu adapter at startup and returns the GPU implementation if available, otherwise falls back to CPU. A `GpuContext` can be pre-created and shared across multiple blur jobs via `create_blurrer_with_context()`.
