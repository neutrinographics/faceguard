use std::sync::{Arc, Mutex};

use wgpu::{self};

/// Descriptor for a single region to blur in a batch.
pub struct RoiDescriptor {
    pub pixels: Vec<u32>,
    pub width: u32,
    pub height: u32,
    pub kernel_size: u32,
    pub ellipse_cx: f32,
    pub ellipse_cy: f32,
    pub ellipse_a: f32,
    pub ellipse_b: f32,
    pub use_ellipse: bool,
}

/// Shared GPU context for blur operations.
///
/// Holds the wgpu device, queue, shader module, and pipeline so they
/// can be reused across frames without re-initialization. GPU buffers
/// are cached internally and reused across `blur_roi()` calls.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub shader: wgpu::ShaderModule,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Interior-mutable buffer cache. Mutex is always uncontended
    /// because blur() is called from a single thread per blurrer instance.
    buffers: Mutex<CachedBuffers>,
}

/// Packed params matching the WGSL uniform layout (48 bytes, 12 x u32).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBlurParams {
    pub width: u32,
    pub height: u32,
    pub kernel_radius: u32,
    pub sigma: f32,
    pub ellipse_cx: f32,
    pub ellipse_cy: f32,
    pub ellipse_a: f32,
    pub ellipse_b: f32,
    pub use_ellipse: u32,
    pub direction: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Pre-allocated GPU buffers reused across blur_roi() calls.
///
/// Buffers are sized to `capacity` pixels. When a larger ROI arrives,
/// all four pixel buffers are reallocated (grow-only, never shrink).
/// Params buffers are fixed at 48 bytes and never reallocated.
struct CachedBuffers {
    capacity: usize,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    original: wgpu::Buffer,
    staging: wgpu::Buffer,
    params_h: wgpu::Buffer,
    params_v: wgpu::Buffer,
    /// Large staging buffer for batch readback (sum of all ROI sizes).
    batch_staging: Option<wgpu::Buffer>,
    batch_staging_capacity: usize,
}

const INITIAL_CAPACITY: usize = 512 * 512;

fn make_pixel_buffers(
    device: &wgpu::Device,
    capacity: usize,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let size = (capacity * 4) as u64;
    let input = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cached-input"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cached-output"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let original = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cached-original"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cached-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (input, output, original, staging)
}

impl GpuContext {
    /// Create a new GPU context. Returns `None` if no suitable adapter is available.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("blur-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gaussian-blur-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gaussian_blur.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur-bind-group-layout"),
            entries: &[
                // params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // input storage (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // output storage (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // original storage (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blur-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (input, output, original, staging) = make_pixel_buffers(&device, INITIAL_CAPACITY);

        let params_size = std::mem::size_of::<GpuBlurParams>() as u64;
        let params_h = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cached-params-h"),
            size: params_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cached-params-v"),
            size: params_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffers = Mutex::new(CachedBuffers {
            capacity: INITIAL_CAPACITY,
            input,
            output,
            original,
            staging,
            params_h,
            params_v,
            batch_staging: None,
            batch_staging_capacity: 0,
        });

        Some(Self {
            device,
            queue,
            shader,
            pipeline,
            bind_group_layout,
            buffers,
        })
    }

    /// Probe for GPU availability without allocating pixel buffers or pipelines.
    pub fn is_available() -> bool {
        let instance = wgpu::Instance::default();
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_some()
    }

    /// Run a two-pass blur on an ROI. Convenience wrapper around `blur_rois`.
    #[allow(clippy::too_many_arguments)]
    pub fn blur_roi(
        &self,
        pixels: &[u32],
        width: u32,
        height: u32,
        kernel_size: u32,
        ellipse_cx: f32,
        ellipse_cy: f32,
        ellipse_a: f32,
        ellipse_b: f32,
        use_ellipse: bool,
    ) -> Vec<u32> {
        let roi = RoiDescriptor {
            pixels: pixels.to_vec(),
            width,
            height,
            kernel_size,
            ellipse_cx,
            ellipse_cy,
            ellipse_a,
            ellipse_b,
            use_ellipse,
        };
        let mut results = self.blur_rois(&[roi]);
        results.remove(0)
    }

    /// Batch-blur multiple ROIs with a single GPU readback.
    ///
    /// Each ROI is processed through the two-pass blur pipeline. All results
    /// are collected into a single staging buffer and read back with one
    /// `device.poll(Wait)` call, eliminating per-region synchronous stalls.
    pub fn blur_rois(&self, rois: &[RoiDescriptor]) -> Vec<Vec<u32>> {
        if rois.is_empty() {
            return vec![];
        }

        let mut cache = self.buffers.lock().unwrap();

        // Find max ROI size and total staging bytes needed.
        let max_pixels = rois
            .iter()
            .map(|r| (r.width * r.height) as usize)
            .max()
            .unwrap_or(0);
        let total_staging_bytes: u64 = rois
            .iter()
            .map(|r| (r.width as u64) * (r.height as u64) * 4)
            .sum();

        // Grow pixel buffers if needed (never shrink).
        if max_pixels > cache.capacity {
            let (inp, out, orig, stg) = make_pixel_buffers(&self.device, max_pixels);
            cache.input = inp;
            cache.output = out;
            cache.original = orig;
            cache.staging = stg;
            cache.capacity = max_pixels;
        }

        // Grow batch staging buffer if needed.
        let total_staging_usize = total_staging_bytes as usize;
        if total_staging_usize > cache.batch_staging_capacity || cache.batch_staging.is_none() {
            cache.batch_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch-staging"),
                size: total_staging_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            cache.batch_staging_capacity = total_staging_usize;
        }

        // Process each ROI: upload, encode, submit (but don't poll yet).
        let mut offsets: Vec<(u64, usize)> = Vec::with_capacity(rois.len());
        let mut staging_offset: u64 = 0;

        for roi in rois {
            let pixel_count = (roi.width * roi.height) as usize;
            let buf_size = (pixel_count * 4) as u64;
            let kernel_radius = roi.kernel_size / 2;
            let sigma = roi.kernel_size as f32 / 6.0;

            // Upload pixel data
            self.queue
                .write_buffer(&cache.input, 0, bytemuck::cast_slice(&roi.pixels));
            self.queue
                .write_buffer(&cache.original, 0, bytemuck::cast_slice(&roi.pixels));

            // Write params
            let params_h = GpuBlurParams {
                width: roi.width,
                height: roi.height,
                kernel_radius,
                sigma,
                ellipse_cx: roi.ellipse_cx,
                ellipse_cy: roi.ellipse_cy,
                ellipse_a: roi.ellipse_a,
                ellipse_b: roi.ellipse_b,
                use_ellipse: if roi.use_ellipse { 1 } else { 0 },
                direction: 0,
                _pad0: 0,
                _pad1: 0,
            };
            self.queue
                .write_buffer(&cache.params_h, 0, bytemuck::bytes_of(&params_h));

            let params_v = GpuBlurParams {
                direction: 1,
                ..params_h
            };
            self.queue
                .write_buffer(&cache.params_v, 0, bytemuck::bytes_of(&params_v));

            // Bind groups
            let bg_h = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg-h"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cache.params_h.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache.input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cache.output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cache.original.as_entire_binding(),
                    },
                ],
            });

            let bg_v = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg-v"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cache.params_v.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache.input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cache.output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cache.original.as_entire_binding(),
                    },
                ],
            });

            let workgroups_x = roi.width.div_ceil(16);
            let workgroups_y = roi.height.div_ceil(16);

            // Encode blur passes + copy to batch staging
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("blur-encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("horizontal"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg_h, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            encoder.copy_buffer_to_buffer(&cache.output, 0, &cache.input, 0, buf_size);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vertical"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg_v, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Copy result to batch staging at this ROI's offset
            encoder.copy_buffer_to_buffer(
                &cache.output,
                0,
                cache.batch_staging.as_ref().unwrap(),
                staging_offset,
                buf_size,
            );

            // Submit this ROI's work (so next ROI's write_buffer won't overwrite)
            self.queue.submit(Some(encoder.finish()));

            offsets.push((staging_offset, pixel_count));
            staging_offset += buf_size;
        }

        // Single synchronous readback for all ROIs
        let batch_buf = cache.batch_staging.as_ref().unwrap();
        let slice = batch_buf.slice(..total_staging_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mapped = slice.get_mapped_range();
        let all_data: &[u32] = bytemuck::cast_slice(&mapped);

        let results: Vec<Vec<u32>> = offsets
            .iter()
            .map(|(offset, count)| {
                let start = (*offset as usize) / 4;
                all_data[start..start + count].to_vec()
            })
            .collect();

        drop(mapped);
        batch_buf.unmap();

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blur_roi_reuses_buffers_same_size() {
        let Some(ctx) = GpuContext::new() else {
            return;
        };
        let pixels: Vec<u32> = vec![0x00FF0000u32; 4 * 4];
        let r1 = ctx.blur_roi(&pixels, 4, 4, 3, 2.0, 2.0, 2.0, 2.0, false);
        let r2 = ctx.blur_roi(&pixels, 4, 4, 3, 2.0, 2.0, 2.0, 2.0, false);
        assert_eq!(r1, r2, "identical inputs must produce identical outputs");
    }

    #[test]
    fn test_blur_roi_grows_buffers_on_larger_roi() {
        let Some(ctx) = GpuContext::new() else {
            return;
        };
        let small: Vec<u32> = vec![0xFFFFFFFFu32; 2 * 2];
        ctx.blur_roi(&small, 2, 2, 3, 1.0, 1.0, 1.0, 1.0, false);
        let large: Vec<u32> = vec![0x00000000u32; 8 * 8];
        let result = ctx.blur_roi(&large, 8, 8, 3, 4.0, 4.0, 4.0, 4.0, false);
        assert_eq!(result.len(), 8 * 8);
    }

    #[test]
    fn test_blur_rois_batch_matches_individual() {
        let Some(ctx) = GpuContext::new() else {
            return;
        };
        let pixels_a: Vec<u32> = vec![0x00FF0000u32; 4 * 4];
        let pixels_b: Vec<u32> = vec![0x0000FF00u32; 6 * 6];

        // Individual calls
        let single_a = ctx.blur_roi(&pixels_a, 4, 4, 3, 2.0, 2.0, 2.0, 2.0, false);
        let single_b = ctx.blur_roi(&pixels_b, 6, 6, 3, 3.0, 3.0, 3.0, 3.0, false);

        // Batch call
        let rois = vec![
            RoiDescriptor {
                pixels: pixels_a,
                width: 4,
                height: 4,
                kernel_size: 3,
                ellipse_cx: 2.0,
                ellipse_cy: 2.0,
                ellipse_a: 2.0,
                ellipse_b: 2.0,
                use_ellipse: false,
            },
            RoiDescriptor {
                pixels: pixels_b,
                width: 6,
                height: 6,
                kernel_size: 3,
                ellipse_cx: 3.0,
                ellipse_cy: 3.0,
                ellipse_a: 3.0,
                ellipse_b: 3.0,
                use_ellipse: false,
            },
        ];
        let batch = ctx.blur_rois(&rois);

        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], single_a, "batch ROI 0 must match individual");
        assert_eq!(batch[1], single_b, "batch ROI 1 must match individual");
    }
}
