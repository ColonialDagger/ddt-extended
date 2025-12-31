import cv2
import numpy as np


def _gmm_mask_from_params(ref, c0, c1):
    """
    Vectorized GMM classifier for 2‑channel input (u*, v*).
    """

    # Stack into (H, W, 2)
    x = np.stack([c0, c1], axis=-1).astype(np.float64)

    weights = np.asarray(ref["weights"], dtype=np.float64)
    means = np.asarray(ref["means"], dtype=np.float64)
    prec_chol = np.asarray(ref["prec_chol"], dtype=np.float64)
    log_dets = np.asarray(ref["log_dets"], dtype=np.float64)
    threshold = float(ref["threshold"])

    log_prob = np.full(x.shape[:2], -np.inf, dtype=np.float64)

    for w, mu, L, log_det in zip(weights, means, prec_chol, log_dets):
        diff = x - mu  # (H, W, 2)

        # 2D Cholesky transform
        y0 = L[0, 0] * diff[..., 0] + L[0, 1] * diff[..., 1]
        y1 = L[1, 0] * diff[..., 0] + L[1, 1] * diff[..., 1]

        md = y0 * y0 + y1 * y1

        lp = (
            -0.5 * md
            + np.log(w)
            - 1.0 * np.log(2 * np.pi)  # 2D Gaussian → -1 * log(2π)
            + log_det
        )

        log_prob = np.logaddexp(log_prob, lp)

    return log_prob >= threshold


def estimate_health(
        cropped_img,
        neg_mask,
        mask_indices,
        mask_col_ids,
        mask_counts,
        color_reference: dict,
        min_col_fraction: float = 0.60,
        edge_width: int = 2
) -> float:
    """
    Hybrid health estimation for D2 boss health bars (right -> left scan).

    Core idea:

    - The active health bar is always the rightmost column (normal health first, then Final Stand(s)).

    - We scan from right to left and find the first column that is "full enough" as per a given threshold.

    - All columns before that point (to the left) are assumed fully healthy.

    - Around the detected edge, perform refine pixel-by-pixel scanning to count health and not health pixels.

    - Add the count of scanned healthy pixels near the detected edge to the count of assumed health pixels.

    :param cropped_img: np.ndarray (H, W, 3)
        A cropped image of the D2 health bar only.
    :param neg_mask: np.ndarray (H, W)
        Boolean mask of valid health bar pixels.
    :param mask_indices: tuple(np.ndarray, np.ndarray)
        Precomputed (ys, xs) for all True pixels in neg_mask.
    :param mask_col_ids: np.ndarray
        Column index for each mask pixel (same length as mask_indices[0]).
    :param mask_counts: np.ndarray
        Precomputed number of mask pixels per column.
    :param min_col_fraction: float
        The minimum number of pixels in a col. to be "healthy" before scanning the next col.
    :param edge_width: int
        The width from the last healthy column that should be scanned in each direction.
    :return:
    """

    h, w, _ = cropped_img.shape

    # Convert to LUV
    luv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LUV).astype(np.float32)

    # Extract u*, v* only
    u = luv[..., 1]
    v = luv[..., 2]

    # Pass u, v into the GMM
    gmm_mask = _gmm_mask_from_params(color_reference, u, v)

    # Healthy = in-range AND inside mask
    # Instead of scanning the whole 2D array, index only mask pixels
    ys, xs = mask_indices
    healthy_flat = gmm_mask[ys, xs]

    # Count healthy pixels per column using bincount
    healthy_counts = np.bincount(
        mask_col_ids,
        weights=healthy_flat.astype(np.int32),
        minlength=w
    )

    # Precompute thresholds
    thresholds = mask_counts * min_col_fraction

    # Local bindings for speed
    hc = healthy_counts
    thr = thresholds
    mc = mask_counts

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
