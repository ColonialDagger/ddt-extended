import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import json
import glob
import os
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DATA_FOLDER = "data"
LUT_FOLDER = "luts"
N_COMPONENTS = 5        # You can increase to 5 or 8 if needed
KEEP_FRACTION = 0.999   # Keep 99.9% of points
COV_TYPE = "full"       # "full" or "tied" recommended

os.makedirs(LUT_FOLDER, exist_ok=True)

# -----------------------------
# GPU-style color pipeline
# -----------------------------

def gpu_bgr_to_luv(bgr: np.ndarray) -> np.ndarray:
    """
    Python version of the WGSL bgr_to_luv() used on GPU.

    bgr: (..., 3) uint8, BGR order
    returns: (..., 3) float32, (L*, u*, v*)
    """
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)

    # sRGB -> linear
    def srgb_to_linear(c: np.ndarray) -> np.ndarray:
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    # RGB -> XYZ (D65)
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


def gpu_uv_to_index(u_star: np.ndarray, v_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map (u*, v*) to [0,255] indices using the same ranges
    as the GPU WGSL shader.
    """
    # These must match WGSL luv_to_uv_index()
    u_min, u_max = -200.0, 300.0
    v_min, v_max = -200.0, 300.0

    u_norm = np.clip((u_star - u_min) / (u_max - u_min), 0.0, 1.0)
    v_norm = np.clip((v_star - v_min) / (v_max - v_min), 0.0, 1.0)

    u_idx = (u_norm * 255.0).astype(np.uint8)
    v_idx = (v_norm * 255.0).astype(np.uint8)
    return u_idx, v_idx

# -----------------------------
# LUT generation from GMM
# -----------------------------

def generate_lut_from_gmm(gmm: GaussianMixture, threshold: float, colorblind_name: str):
    """
    Evaluate the GMM over a 256x256 UV grid (GPU-style indices) and produce a LUT.
    Save as a binary blob and a PNG visualization.
    """
    # Create a 256x256 grid of (u_idx, v_idx) values
    u = np.arange(256, dtype=np.float64)
    v = np.arange(256, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uv_points = np.stack([uu, vv], axis=-1).reshape(-1, 2)  # shape (65536, 2)

    # Evaluate GMM log-probabilities
    log_probs = gmm.score_samples(uv_points)

    lut_flat = (log_probs >= threshold).astype(np.uint32)
    lut = lut_flat.reshape(256, 256)

    # Save binary LUT
    bin_path = os.path.join(LUT_FOLDER, f"lut_{colorblind_name}.bin")
    with open(bin_path, "wb") as f:
        f.write(lut.tobytes())  # now 65536 * 4 bytes

    # Optional: save visualization
    png_path = os.path.join(LUT_FOLDER, f"lut_{colorblind_name}.png")
    plt.imshow(lut, cmap="gray", origin="lower")
    plt.title(f"LUT for {colorblind_name}")
    plt.colorbar()
    plt.savefig(png_path)
    plt.close()


def train_gmm_for_file(csv_path: str):
    """Load CSV, convert with GPU-style LUV, train GMM in UV-index space."""
    df = pd.read_csv(csv_path, header=None)

    # CSV is BGR rows; shape (N, 3)
    rgb = df.to_numpy(dtype=np.uint8)  # still BGR actually
    # Reshape to (N, 1, 3) to mimic image-ish shape for our function
    bgr_img = rgb.reshape(-1, 1, 3)

    # GPU-style LUV
    luv = gpu_bgr_to_luv(bgr_img).reshape(-1, 3).astype(np.float64)
    u_star = luv[:, 1]
    v_star = luv[:, 2]

    # Map to GPU UV index space [0,255]
    u_idx, v_idx = gpu_uv_to_index(u_star, v_star)
    uv = np.stack([u_idx.astype(np.float64), v_idx.astype(np.float64)], axis=1)

    # Chroma magnitude in this UV-index space
    uv_norm = np.sqrt(u_idx.astype(np.float64)**2 + v_idx.astype(np.float64)**2)

    # Keep only "strong chroma" pixels (tuned threshold)
    mask = uv_norm >= 15.0
    points = uv[mask]

    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type=COV_TYPE,
    )
    gmm.fit(points)

    # Compute log-probability threshold
    scores = gmm.score_samples(points)
    threshold = np.quantile(scores, 1 - KEEP_FRACTION)

    # Extract parameters for JSON (still useful for CPU/GPU debug)
    weights = gmm.weights_.tolist()
    means = gmm.means_.tolist()
    prec_chol = gmm.precisions_cholesky_.tolist()

    # Compute log-determinants from Cholesky precision
    log_dets = []
    for L in gmm.precisions_cholesky_:
        log_det = -2.0 * float(np.sum(np.log(np.diag(L))))
        log_dets.append(log_det)

    params = {
        "weights": weights,
        "means": means,
        "prec_chol": prec_chol,
        "log_dets": log_dets,
        "threshold": float(threshold),
    }

    return params, gmm, threshold


def main():
    # Find all colorblind-mode CSVs
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "pixels_*.csv")))

    color_refs = {}

    for f in files:
        base = os.path.basename(f)
        colorblind = base.split("_")[1].split(".")[0]

        print(f"Training colorblind mode {colorblind} from {base}... ", end="")
        params, gmm, threshold = train_gmm_for_file(f)
        color_refs[colorblind] = params
        print("DONE")

        print(f"Generating LUT for {colorblind}... ", end="")
        generate_lut_from_gmm(gmm, threshold, colorblind)
        print("DONE")

    # Save color refs JSON too (if you still use it elsewhere)
    json_path = os.path.join(LUT_FOLDER, "color_refs_gpu.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(color_refs, f, indent=2)


if __name__ == "__main__":
    main()
