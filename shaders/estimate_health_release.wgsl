struct ScanParams {
    width: u32,
    height: u32,
};

@group(0) @binding(0)
var<storage, read> pixels: array<u32>;   // packed BGRA8 0xAARRGGBB

@group(0) @binding(1)
var<storage, read> neg_mask: array<u32>; // 0 or 1 per pixel

@group(0) @binding(2)
var<storage, read> lut: array<u32>;      // 0/1 per entry, 256*256

@group(0) @binding(3)
var<storage, read_write> stats: array<atomic<u32>>;
// stats[0] = total_healthy, stats[1] = total_mask

@group(0) @binding(4)
var<uniform> params: ScanParams;

// ---------------------------------
// Helpers
// ---------------------------------
fn unpack_bgr8(packed: u32) -> vec3<u32> {
    let b = packed & 0xFFu;
    let g = (packed >> 8u) & 0xFFu;
    let r = (packed >> 16u) & 0xFFu;
    return vec3<u32>(b, g, r);
}

fn srgb_to_linear(c: f32) -> f32 {
    let cs = c / 255.0;
    if (cs <= 0.04045) {
        return cs / 12.92;
    }
    return pow((cs + 0.055) / 1.055, 2.4);
}

// BGR (0–255) -> CIE L*u*v*
fn bgr_to_luv(bgr: vec3<u32>) -> vec3<f32> {
    let b_lin = srgb_to_linear(f32(bgr.x));
    let g_lin = srgb_to_linear(f32(bgr.y));
    let r_lin = srgb_to_linear(f32(bgr.z));

    let X = 0.4124 * r_lin + 0.3576 * g_lin + 0.1805 * b_lin;
    let Y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;
    let Z = 0.0193 * r_lin + 0.1192 * g_lin + 0.9505 * b_lin;

    let Xn: f32 = 0.95047;
    let Yn: f32 = 1.0;
    let Zn: f32 = 1.08883;

    let denom = X + 15.0 * Y + 3.0 * Z;
    var u_prime: f32 = 0.0;
    var v_prime: f32 = 0.0;
    if (denom > 1e-9) {
        u_prime = 4.0 * X / denom;
        v_prime = 9.0 * Y / denom;
    }

    let denom_n = Xn + 15.0 * Yn + 3.0 * Zn;
    let u_prime_n = 4.0 * Xn / denom_n;
    let v_prime_n = 9.0 * Yn / denom_n;

    let yr = Y / Yn;
    var L: f32;
    if (yr > 0.008856) {
        L = 116.0 * pow(yr, 1.0 / 3.0) - 16.0;
    } else {
        L = 903.3 * yr;
    }
    if (L < 0.0) {
        L = 0.0;
    }

    let u_star = 13.0 * L * (u_prime - u_prime_n);
    let v_star = 13.0 * L * (v_prime - v_prime_n);

    return vec3<f32>(L, u_star, v_star);
}

// Map u*, v* to [0,255] LUT index space
fn luv_to_uv_index(luv: vec3<f32>) -> vec2<u32> {
    let u_star = luv.y;
    let v_star = luv.z;

    let u_min: f32 = -200.0;
    let u_max: f32 =  300.0;
    let v_min: f32 = -200.0;
    let v_max: f32 =  300.0;

    let u_norm = clamp((u_star - u_min) / (u_max - u_min), 0.0, 1.0);
    let v_norm = clamp((v_star - v_min) / (v_max - v_min), 0.0, 1.0);

    let u_idx = u32(round(u_norm * 255.0));
    let v_idx = u32(round(v_norm * 255.0));

    return vec2<u32>(u_idx, v_idx);
}

// ---------------------------------
// Single pass: classify + total counts
// ---------------------------------
@compute @workgroup_size(16, 16)
fn classify_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width: u32 = params.width;
    let height: u32 = params.height;

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let idx: u32 = gid.y * width + gid.x;

    let mask_val: u32 = neg_mask[idx];
    if (mask_val == 0u) {
        return;
    }

    let packed: u32 = pixels[idx];
    let bgr: vec3<u32> = unpack_bgr8(packed);
    let luv: vec3<f32> = bgr_to_luv(bgr);
    let uv_idx: vec2<u32> = luv_to_uv_index(luv);

    let u: u32 = clamp(uv_idx.x, 0u, 255u);
    let v: u32 = clamp(uv_idx.y, 0u, 255u);

    let lut_idx: u32 = u * 256u + v;
    let healthy_u32: u32 = lut[lut_idx] & 1u;

    // stats[0] = total_healthy, stats[1] = total_mask
    atomicAdd(&stats[1u], 1u); // total_mask++
    if (healthy_u32 == 1u) {
        atomicAdd(&stats[0u], 1u); // total_healthy++
    }
}
