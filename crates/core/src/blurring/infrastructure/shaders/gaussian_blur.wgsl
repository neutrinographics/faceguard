// Two-direction separable Gaussian blur with optional ellipse masking.
//
// Pass 0 = horizontal blur, Pass 1 = vertical blur + ellipse composite.
// The original (unblurred) image is kept in `original` for the final
// masked composite step.

struct Params {
    width: u32,
    height: u32,
    kernel_radius: u32,  // half-size of kernel (kernel_size = 2*radius + 1)
    sigma: f32,
    // Ellipse mask parameters (only used in vertical direction)
    ellipse_cx: f32,
    ellipse_cy: f32,
    ellipse_a: f32,   // semi-axis x
    ellipse_b: f32,   // semi-axis y
    use_ellipse: u32, // 0 = rectangular (no mask), 1 = elliptical
    direction: u32,        // 0 = horizontal, 1 = vertical
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> original: array<u32>;

fn unpack_rgba(packed: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(packed & 0xFFu),
        f32((packed >> 8u) & 0xFFu),
        f32((packed >> 16u) & 0xFFu),
        f32((packed >> 24u) & 0xFFu),
    );
}

fn pack_rgba(v: vec4<f32>) -> u32 {
    let r = u32(clamp(v.x, 0.0, 255.0));
    let g = u32(clamp(v.y, 0.0, 255.0));
    let b = u32(clamp(v.z, 0.0, 255.0));
    let a = u32(clamp(v.w, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let idx = y * params.width + x;
    var color = vec4<f32>(0.0);
    var weight_sum = 0.0;

    let radius = i32(params.kernel_radius);
    let sigma2 = 2.0 * params.sigma * params.sigma;

    if params.direction == 0u {
        // Horizontal direction
        for (var k = -radius; k <= radius; k = k + 1) {
            let sx = clamp(i32(x) + k, 0, i32(params.width) - 1);
            let sample_idx = y * params.width + u32(sx);
            let w = exp(-f32(k * k) / sigma2);
            color += unpack_rgba(input[sample_idx]) * w;
            weight_sum += w;
        }
    } else {
        // Vertical direction
        for (var k = -radius; k <= radius; k = k + 1) {
            let sy = clamp(i32(y) + k, 0, i32(params.height) - 1);
            let sample_idx = u32(sy) * params.width + x;
            let w = exp(-f32(k * k) / sigma2);
            color += unpack_rgba(input[sample_idx]) * w;
            weight_sum += w;
        }
    }

    var blurred = color / weight_sum;

    // Apply ellipse mask on vertical direction (final output)
    if params.direction == 1u && params.use_ellipse == 1u {
        let dx = f32(x) - params.ellipse_cx;
        let dy = f32(y) - params.ellipse_cy;
        var dist = 999.0;
        if params.ellipse_a > 0.0 && params.ellipse_b > 0.0 {
            dist = (dx / params.ellipse_a) * (dx / params.ellipse_a) +
                   (dy / params.ellipse_b) * (dy / params.ellipse_b);
        }
        if dist > 1.0 {
            // Outside ellipse: use original pixel
            blurred = unpack_rgba(original[idx]);
        }
    }

    output[idx] = pack_rgba(blurred);
}
