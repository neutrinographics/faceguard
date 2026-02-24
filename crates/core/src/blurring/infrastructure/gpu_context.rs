use std::sync::{Arc, Mutex};

use wgpu::{self};

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

    /// Run a two-pass blur on an ROI. `pixels` is packed RGBA u32 data.
    /// Returns the blurred pixel data as packed RGBA u32.
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
        let pixel_count = (width * height) as usize;
        let buf_size = (pixel_count * 4) as u64;
        let kernel_radius = kernel_size / 2;
        let sigma = kernel_size as f32 / 6.0;

        let mut cache = self.buffers.lock().unwrap();

        // Grow buffers if needed (never shrink).
        if pixel_count > cache.capacity {
            let (inp, out, orig, stg) = make_pixel_buffers(&self.device, pixel_count);
            cache.input = inp;
            cache.output = out;
            cache.original = orig;
            cache.staging = stg;
            cache.capacity = pixel_count;
        }

        // Upload pixel data
        self.queue
            .write_buffer(&cache.input, 0, bytemuck::cast_slice(pixels));
        self.queue
            .write_buffer(&cache.original, 0, bytemuck::cast_slice(pixels));

        // Write params for both passes
        let params_h = GpuBlurParams {
            width,
            height,
            kernel_radius,
            sigma,
            ellipse_cx,
            ellipse_cy,
            ellipse_a,
            ellipse_b,
            use_ellipse: if use_ellipse { 1 } else { 0 },
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

        // Bind groups for each pass
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

        let workgroups_x = width.div_ceil(16);
        let workgroups_y = height.div_ceil(16);

        // Single encoder: pass0 → copy(output→input) → pass1 → copy(output→staging)
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

        encoder.copy_buffer_to_buffer(&cache.output, 0, &cache.staging, 0, buf_size);

        self.queue.submit(Some(encoder.finish()));

        // Synchronous readback
        let slice = cache.staging.slice(..buf_size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mapped = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        cache.staging.unmap();

        result
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
}
