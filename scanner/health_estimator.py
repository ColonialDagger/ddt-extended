import cv2
import numpy as np

class HealthReference:
    neg_mask : np.ndarray = None
    mask_indices = None
    mask_col_ids = None
    mask_counts = None
    lut = None

def _bgr_to_luv(bgr: np.ndarray) -> np.ndarray:
    """
    Converts image from BGR to LUV color space.

    :param bgr:
    :return:
    """
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)

    def srgb_to_linear(c: np.ndarray) -> np.ndarray:
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    X = 0.4124 * r_lin + 0.3576 * g_lin + 0.1805 * b_lin
    Y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    Z = 0.0193 * r_lin + 0.1192 * g_lin + 0.9505 * b_lin

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    denom = X + 15.0 * Y + 3.0 * Z
    safe_denom = np.where(denom > 1e-9, denom, 1.0)

    u_prime = 4.0 * X / safe_denom
    v_prime = 9.0 * Y / safe_denom

    denom_n = Xn + 15.0 * Yn + 3.0 * Zn
    u_prime_n = 4.0 * Xn / denom_n
    v_prime_n = 9.0 * Yn / denom_n

    yr = Y / Yn
    L = np.where(yr > 0.008856, 116.0 * np.cbrt(yr) - 16.0, 903.3 * yr)
    L = np.maximum(L, 0.0)

    u_star = 13.0 * L * (u_prime - u_prime_n)
    v_star = 13.0 * L * (v_prime - v_prime_n)

    return np.stack([L, u_star, v_star], axis=-1).astype(np.float32)


def _gpu_uv_to_index(u_star: np.ndarray, v_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Maps (u*, v*) → [0,255] UV-index space used by the GPU shader. This will match future compute shaders, assuming I
    can ever get compilation to work for it.

    Args:
        u_star: array of u* values.
        v_star: array of v* values.

    Returns:
        (u_idx, v_idx): uint8 arrays in [0,255].
    """
    # These ranges must match the GPU WGSL shader
    u_min, u_max = -200.0, 300.0
    v_min, v_max = -200.0, 300.0

    u_norm = np.clip((u_star - u_min) / (u_max - u_min), 0.0, 1.0)
    v_norm = np.clip((v_star - v_min) / (v_max - v_min), 0.0, 1.0)

    return (u_norm * 255.0).astype(np.uint8), (v_norm * 255.0).astype(np.uint8)


def estimate_health(
        cropped_img,
        ref,
        min_col_fraction: float = 0.60,
        edge_width: int = 2
) -> float:
    """
    Hybrid health estimation for D2 boss health bars (right -> left scan).

    This LUT-powered version replaces the GMM classifier entirely.
    Classification is now a single LUT lookup per pixel, matching the GPU shader.

    Core idea:

    - The active health bar is always the rightmost column (normal health first, then Final Stand(s)).

    - We scan from right to left and find the first column that is "full enough" as per a given threshold.

    - All columns before that point (to the left) are assumed fully healthy.

    - Around the detected edge, perform refine pixel-by-pixel scanning to count health and not health pixels.

    - Add the count of scanned healthy pixels near the detected edge to the count of assumed health pixels.

    :param cropped_img: np.ndarray (H, W, 3)
        A cropped image of the D2 health bar only.
    :param ref: HealthReference
        Contains reference data for health estimation. This is generated every time the scanner initializes.
    :param min_col_fraction: float
        The minimum number of pixels in a col. to be "healthy" before scanning the next col.
    :param edge_width: int
        The width from the last healthy column that should be scanned in each direction.
    :return:
    """

    h, w, _ = cropped_img.shape

    # Convert to LUV
    luv = _bgr_to_luv(cropped_img)
    u_star = luv[..., 1]
    v_star = luv[..., 2]

    # Map to UV-index space
    u_idx, v_idx = _gpu_uv_to_index(u_star, v_star)

    # LUT classification
    lut_mask = ref.lut[u_idx, v_idx]          # shape (H, W), values 0 or 1

    # Healthy = LUT AND neg_mask
    healthy_mask = np.bitwise_and(lut_mask, ref.neg_mask)

    # Flattened healthy pixels at mask locations
    ys, xs = ref.mask_indices
    healthy_flat = healthy_mask[ys, xs]

    # Count healthy pixels per column
    healthy_counts = np.bincount(
        ref.mask_col_ids,
        weights=healthy_flat.astype(np.int32),
        minlength=w
    )

    # Precompute thresholds
    thresholds = ref.mask_counts * min_col_fraction

    # Local bindings for speed
    hc = healthy_counts
    thr = thresholds
    mc = ref.mask_counts

    # Step 1.1: Fast full-health early exit
    last_col = w - 1
    if mc[last_col] > 0 and hc[last_col] >= thr[last_col]:
        first_full_col = last_col
    else:
        # Step 1.2: Scan right → left for first full column
        first_full_col = -1
        for col in range(last_col, -1, -1):
            if mc[col] == 0:
                continue
            if hc[col] >= thr[col]:
                first_full_col = col
                break

    # Step 1.5: Fallback if no full column found
    if first_full_col < 0:
        total = mc[last_col]
        if total == 0:
            return 0.0
        return hc[last_col] / total

    # Step 2: Count all mask pixels BEFORE the first full column
    if first_full_col > 0:
        full_columns_count = mc[:first_full_col].sum()
    else:
        full_columns_count = 0

    # Step 3: Pixel-level refinement around the edge
    edge_start = max(0, first_full_col - edge_width)
    edge_end = min(w, first_full_col + edge_width + 1)

    edge_healthy = hc[edge_start:edge_end].sum()

    # Step 4: Combine counts
    total_bar_pixels = mc.sum()
    total_healthy = full_columns_count + edge_healthy

    health_fraction = total_healthy / total_bar_pixels
    return min(1.0, max(0.0, health_fraction))
