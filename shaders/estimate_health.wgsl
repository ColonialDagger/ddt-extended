struct ScanParams {
    width: u32,
    height: u32,
    full_column_min_fraction: f32,
    edge_width: u32,
};

@group(0) @binding(0)
var<storage, read> pixels: array<u32>;   // packed BGRA8 0xAARRGGBB

@group(0) @binding(1)
var<storage, read> neg_mask: array<u32>; // 0 or 1 per pixel

@group(0) @binding(2)
var<storage, read> lut: array<u32>;      // 0/1 per entry, 256*256

@group(0) @binding(3)
var<storage, read_write> stats: array<atomic<u32>>;

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

// BGR (0–255) -> CIE L*u*v* (matches generator)
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

// Map u*, v* to [0,255] index space (matches generator)
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

fn idx_total_healthy(width: u32) -> u32 { return 2u * width + 0u; }
fn idx_total_mask(width: u32)    -> u32 { return 2u * width + 1u; }
fn idx_first_full(width: u32)    -> u32 { return 2u * width + 2u; }
fn idx_full_cols(width: u32)     -> u32 { return 2u * width + 3u; }
fn idx_edge_start(width: u32)    -> u32 { return 2u * width + 4u; }
fn idx_edge_end(width: u32)      -> u32 { return 2u * width + 5u; }
fn idx_final_health(width: u32)  -> u32 { return 2u * width + 6u; }

// ---------------------------------
// PASS 1: classification + stats
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

    let col: u32 = gid.x;

    let healthy_col_offset: u32 = col;
    let mask_col_offset: u32 = width + col;
    let total_healthy_idx: u32 = idx_total_healthy(width);
    let total_mask_idx: u32    = idx_total_mask(width);

    atomicAdd(&stats[mask_col_offset], 1u);
    if (healthy_u32 == 1u) {
        atomicAdd(&stats[healthy_col_offset], 1u);
        atomicAdd(&stats[total_healthy_idx], 1u);
    }
    atomicAdd(&stats[total_mask_idx], 1u);
}

// ---------------------------------
// PASS 2: reduction (total_healthy / total_mask)
// ---------------------------------
@compute @workgroup_size(1, 1, 1)
fn reduce_health(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    let width: u32 = params.width;

    let total_healthy_idx: u32 = idx_total_healthy(width);
    let total_mask_idx: u32    = idx_total_mask(width);

    let th: u32 = atomicLoad(&stats[total_healthy_idx]);
    let tm: u32 = atomicLoad(&stats[total_mask_idx]);

    // Debug slots: mark no first_full_col yet, zero everything else
    atomicStore(&stats[idx_first_full(width)], 0xFFFFFFFFu);
    atomicStore(&stats[idx_full_cols(width)], 0u);
    atomicStore(&stats[idx_edge_start(width)], 0u);
    atomicStore(&stats[idx_edge_end(width)], 0u);

    var health_f: f32 = 0.0;
    if (tm > 0u) {
        health_f = f32(th) / f32(tm);
    } else {
        health_f = 0.0;
    }

    if (health_f < 0.0) {
        health_f = 0.0;
    } else if (health_f > 1.0) {
        health_f = 1.0;
    }

    let bits = bitcast<u32>(health_f);
    atomicStore(&stats[idx_final_health(width)], bits);
}
