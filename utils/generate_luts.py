import os
import csv
import numpy as np
from sklearn.mixture import GaussianMixture

DATA_FOLDER = "data"
LUT_FOLDER = "luts"

N_COMPONENTS = 5  # Number of GMM components
KEEP_FRACTION = 0.999  # Keep top 99.9% of log-probability mass
COV_TYPE = "full"  # Recommended: "full" or "tied"

os.makedirs(LUT_FOLDER, exist_ok=True)

def _gpu_bgr_to_luv(bgr: np.ndarray) -> np.ndarray:
    """
    Converts BGR uint8 → L*, u*, v* float32 using the same math as the GPU shader.

    Args:
        bgr: (..., 3) uint8 array in BGR order.

    Returns:
        (..., 3) float32 array of (L*, u*, v*).
    """
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)

    # sRGB → linear
    def srgb_to_linear(c: np.ndarray) -> np.ndarray:
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    # RGB → XYZ (D65)
    X = 0.4124 * r_lin + 0.3576 * g_lin + 0.1805 * b_lin
    Y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    Z = 0.0193 * r_lin + 0.1192 * g_lin + 0.9505 * b_lin

    # Reference white (D65)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    denom = X + 15.0 * Y + 3.0 * Z
    u_prime = np.where(denom > 1e-9, 4.0 * X / denom, 0.0)
    v_prime = np.where(denom > 1e-9, 9.0 * Y / denom, 0.0)

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
    Maps (u*, v*) → [0,255] UV-index space used by the GPU shader.

    Args:
        u_star: array of u* values.
        v_star: array of v* values.

    Returns:
        (u_idx, v_idx): uint8 arrays in [0,255].
    """
    u_min, u_max = -200.0, 300.0
    v_min, v_max = -200.0, 300.0

    u_norm = np.clip((u_star - u_min) / (u_max - u_min), 0.0, 1.0)
    v_norm = np.clip((v_star - v_min) / (v_max - v_min), 0.0, 1.0)

    return (u_norm * 255.0).astype(np.uint8), (v_norm * 255.0).astype(np.uint8)

def _train_gmm(csv_path: str) -> tuple[GaussianMixture, float]:
    """
    Loads a CSV of BGR pixels, converts to GPU-style UV-index space,
    trains a GMM, and computes the log-probability threshold.

    Args:
        csv_path: Path to CSV file containing BGR rows.

    Returns:
        (gmm, threshold)
    """
    # Load CSV manually (faster than pandas for small files)
    pixels = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            pixels.append([int(row[0]), int(row[1]), int(row[2])])

    bgr = np.asarray(pixels, dtype=np.uint8).reshape(-1, 1, 3)

    # Convert to LUV
    luv = _gpu_bgr_to_luv(bgr).reshape(-1, 3).astype(np.float64)
    u_star = luv[:, 1]
    v_star = luv[:, 2]

    # Map to UV-index space
    u_idx, v_idx = _gpu_uv_to_index(u_star, v_star)
    uv = np.stack([u_idx.astype(np.float64), v_idx.astype(np.float64)], axis=1)

    # Filter low-chroma points
    chroma = np.sqrt(u_idx.astype(np.float64)**2 + v_idx.astype(np.float64)**2)
    points = uv[chroma >= 15.0]

    # Train GMM
    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type=COV_TYPE,
    )
    gmm.fit(points)

    # Compute log-probability threshold
    scores = gmm.score_samples(points)
    threshold = np.quantile(scores, 1 - KEEP_FRACTION)

    return gmm, float(threshold)

def _generate_lut(gmm: GaussianMixture, threshold: float, mode: str) -> None:
    """
    Evaluates the GMM over a 256×256 UV grid and writes a binary LUT.

    Args:
        gmm: Trained GaussianMixture model.
        threshold: Log-probability threshold.
        mode: Colorblind mode name.
    """
    # Build UV grid
    u = np.arange(256, dtype=np.float64)
    v = np.arange(256, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uv_points = np.stack([uu, vv], axis=-1).reshape(-1, 2)

    # Evaluate GMM
    log_probs = gmm.score_samples(uv_points)
    lut = (log_probs >= threshold).astype(np.uint32).reshape(256, 256)

    # Write binary LUT
    out_path = os.path.join(LUT_FOLDER, f"{mode}.bin")
    with open(out_path, "wb") as f:
        f.write(lut.tobytes())

def main() -> None:
    """
    LUT generator for Destiny 2 health-bar color classification.

    This script:
    1. Loads pixel CSVs for each colorblind mode.
    2. Converts BGR → GPU-style LUV.
    3. Maps (u*, v*) → GPU UV-index space [0,255].
    4. Trains a GMM in UV-index space.
    5. Evaluates the GMM over a 256×256 grid.
    6. Writes a binary LUT to luts/{mode}.bin.
    """
    for name in os.listdir(DATA_FOLDER):
        if not name.startswith("pixels_") or not name.endswith(".csv"):
            continue

        mode = name.split("_")[1].split(".")[0]
        csv_path = os.path.join(DATA_FOLDER, name)

        print(f"Training GMM for mode '{mode}'... ", end="")
        gmm, threshold = _train_gmm(csv_path)
        print("DONE")

        print(f"Generating LUT for '{mode}'... ", end="")
        _generate_lut(gmm, threshold, mode)
        print("DONE")


if __name__ == "__main__":
    main()