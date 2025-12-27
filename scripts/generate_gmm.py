import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import json
import glob
import os

DATA_FOLDER = "data"
N_COMPONENTS = 3          # You can increase to 5 or 8 if needed
KEEP_FRACTION = 0.999     # Keep 99.9% of points (looser threshold)
COV_TYPE = "full"         # "tied" or "full" are best for RGB


def train_gmm_for_file(csv_path):
    """Load CSV, train GMM, return parameters."""
    df = pd.read_csv(csv_path, header=None)
    points = df.to_numpy(dtype=np.float64)

    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type=COV_TYPE
    )
    gmm.fit(points)

    # Compute log-probability threshold
    scores = gmm.score_samples(points)
    threshold = np.quantile(scores, 1 - KEEP_FRACTION)

    # Extract parameters
    weights = gmm.weights_.tolist()
    means = gmm.means_.tolist()
    covs = gmm.covariances_.tolist()

    return {
        "weights": weights,
        "means": means,
        "covs": covs,
        "threshold": float(threshold)
    }


def main():
    # Find all brightness CSVs
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "pixels_unique_b*.csv")))

    color_refs = {}

    for f in files:
        # Extract brightness number from filename
        base = os.path.basename(f)
        # pixels_unique_b3.csv → 3
        brightness = int(base.split("_b")[1].split(".")[0])

        print(f"Training brightness {brightness} from {base}...")

        params = train_gmm_for_file(f)
        color_refs[brightness] = params

    # Pretty-print dictionary so you can paste it into Scanner
    print("\n\n==================== COPY BELOW ====================\n")
    print("COLOR_REFS = ", end="")
    print(json.dumps(color_refs, indent=4))
    print("\n==================== COPY ABOVE ====================\n")


if __name__ == "__main__":
    main()
