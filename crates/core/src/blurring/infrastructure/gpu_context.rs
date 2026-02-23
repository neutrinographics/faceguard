use std::sync::Arc;
use wgpu::{self};

/// Shared GPU context for blur operations.
///
/// Holds the wgpu device, queue, shader module, and pipeline so they
/// can be reused across frames without re-initialization.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub shader: wgpu::ShaderModule,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
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

        Some(Self {
            device,
            queue,
            shader,
            pipeline,
            bind_group_layout,
        })
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
        let buf_size = (pixel_count * 4) as u64; // 4 bytes per u32
        let kernel_radius = kernel_size / 2;
        let sigma = kernel_size as f32 / 6.0;

        // Create buffers
        let input_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let original_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("original"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload pixel data
        self.queue
            .write_buffer(&input_buf, 0, bytemuck::cast_slice(pixels));
        self.queue
            .write_buffer(&original_buf, 0, bytemuck::cast_slice(pixels));

        let workgroups_x = width.div_ceil(16);
        let workgroups_y = height.div_ceil(16);

        // Pass 0: horizontal blur (input → output)
        {
            let params = GpuBlurParams {
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
            let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params-h"),
                size: std::mem::size_of::<GpuBlurParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg-h"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: original_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("enc-h"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("horizontal"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        // Copy output → input for second pass
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy"),
                });
            encoder.copy_buffer_to_buffer(&output_buf, 0, &input_buf, 0, buf_size);
            self.queue.submit(Some(encoder.finish()));
        }

        // Pass 1: vertical blur + ellipse mask (input → output)
        {
            let params = GpuBlurParams {
                width,
                height,
                kernel_radius,
                sigma,
                ellipse_cx,
                ellipse_cy,
                ellipse_a,
                ellipse_b,
                use_ellipse: if use_ellipse { 1 } else { 0 },
                direction: 1,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params-v"),
                size: std::mem::size_of::<GpuBlurParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg-v"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: original_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("enc-v"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vertical"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            // Copy result to staging
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, buf_size);
            self.queue.submit(Some(encoder.finish()));
        }

        // Read back results
        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mapped = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging_buf.unmap();

        result
    }
}
