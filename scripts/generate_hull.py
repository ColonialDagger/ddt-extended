import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations

from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA


# ============================================================
# Load CSV
# ============================================================

def load_rgb_csv(path):
    df = pd.read_csv(path)
    if not set(["R", "G", "B"]).issubset(df.columns):
        df.columns = ["R", "G", "B"] + list(df.columns[3:])
    return df[["R", "G", "B"]].to_numpy(dtype=np.float64)


# ============================================================
# Convex Hull Classifier
# ============================================================

def train_convex_hull(points):
    hull = ConvexHull(points)
    A = hull.equations[:, :3]
    b = hull.equations[:, 3]
    return A, b, hull

def classify_convex_hull(points, A, b):
    inside = np.sum((A @ points.T + b[:, None]) <= 0, axis=0) == A.shape[0]
    return inside


# ============================================================
# Ellipsoid (Mahalanobis)
# ============================================================

def train_ellipsoid(points, percentile=0.99):
    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    diffs = points - mean
    m2 = np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)
    k = np.quantile(m2, percentile)

    return mean, cov_inv, k

def classify_ellipsoid(points, mean, cov_inv, k):
    diffs = points - mean
    m2 = np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)
    return m2 <= k


# ============================================================
# PCA Oriented Bounding Box (OBB)
# ============================================================

def train_pca_obb(points):
    pca = PCA(n_components=3)
    pts_pca = pca.fit_transform(points)
    mins = pts_pca.min(axis=0)
    maxs = pts_pca.max(axis=0)
    return pca, mins, maxs

def classify_pca_obb(points, pca, mins, maxs):
    pts_pca = pca.transform(points)
    inside = np.all((pts_pca >= mins) & (pts_pca <= maxs), axis=1)
    return inside


# ============================================================
# Gaussian Mixture Model (GMM)
# ============================================================

def train_gmm(points, n_components=3):
    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.fit(points)
    scores = gmm.score_samples(points)
    threshold = np.quantile(scores, 0.001)
    weights = gmm.weights_
    means = gmm.means_
    covs = gmm.covariances_
    print("weights =", weights.tolist())
    print("means =", means.tolist())
    print("covs =", covs.tolist())
    print("threshold =", float(threshold))

    return gmm, threshold

def classify_gmm(points, gmm, threshold):
    scores = gmm.score_samples(points)
    return scores >= threshold


# ============================================================
# One-Class SVM
# ============================================================

def train_ocsvm(points, nu=0.01):
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    model.fit(points)
    return (model,)  # must return tuple

def classify_ocsvm(points, model):
    preds = model.predict(points)
    return preds == 1


# ============================================================
# Visualization Helpers
# ============================================================

def plot_point_cloud(ax, points):
    ax.scatter(points[:,0], points[:,1], points[:,2],
               c=points/255.0, s=8, alpha=0.4)

def plot_convex_hull(ax, points, hull):
    for simplex in hull.simplices:
        tri = points[simplex]
        poly = Poly3DCollection([tri], alpha=0.15)
        poly.set_facecolor("cyan")
        ax.add_collection3d(poly)

def sample_ellipsoid(mean, cov_inv, k, n=40):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    pts = []

    L = np.linalg.cholesky(np.linalg.inv(cov_inv))

    for ui in u:
        for vi in v:
            sphere = np.array([
                np.cos(ui) * np.sin(vi),
                np.sin(ui) * np.sin(vi),
                np.cos(vi)
            ])
            x = mean + L @ sphere * np.sqrt(k)
            pts.append(x)

    return np.array(pts)

def plot_ellipsoid(ax, ellipsoid_pts):
    ax.scatter(ellipsoid_pts[:,0], ellipsoid_pts[:,1], ellipsoid_pts[:,2],
               color="red", s=3, alpha=0.3)

def obb_corners(pca, mins, maxs):
    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]],
        [maxs[0], maxs[1], maxs[2]],
    ])
    return pca.inverse_transform(corners)

def plot_obb(ax, obb_pts):
    for i, j in combinations(range(8), 2):
        if np.sum(np.abs(obb_pts[i] - obb_pts[j]) < 1e-6) == 2:
            ax.plot(*zip(obb_pts[i], obb_pts[j]), color="green")

def plot_gmm_region(ax, gmm, threshold, resolution=20):
    grid = np.linspace(0, 255, resolution)
    X, Y, Z = np.meshgrid(grid, grid, grid)
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    scores = gmm.score_samples(pts)
    inside = pts[scores >= threshold]

    ax.scatter(inside[:,0], inside[:,1], inside[:,2],
               color="purple", s=2, alpha=0.2)

def plot_ocsvm_region(ax, model, resolution=20):
    grid = np.linspace(0, 255, resolution)
    X, Y, Z = np.meshgrid(grid, grid, grid)
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    preds = model.predict(pts)
    inside = pts[preds == 1]

    ax.scatter(inside[:,0], inside[:,1], inside[:,2],
               color="orange", s=2, alpha=0.2)


# ============================================================
# Benchmark Helper
# ============================================================

def benchmark(name, train_fn, classify_fn, points, *args):
    print(f"\n=== {name} ===")

    t0 = time.time()
    params = train_fn(points, *args)
    t_train = time.time() - t0

    t1 = time.time()
    inside = classify_fn(points, *params)
    t_test = time.time() - t1

    inside_count = np.sum(inside)
    outside_count = len(points) - inside_count

    print(f"Train time: {t_train:.4f}s")
    print(f"Test time:  {t_test:.4f}s")
    print(f"Inside:     {inside_count}")
    print(f"Outside:    {outside_count}")

    return params


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    csv_path = "data/pixels_unique_b4.csv"
    points = load_rgb_csv(csv_path)

    print(f"Loaded {len(points)} RGB samples")

    # Train all classifiers
    A, b, hull = benchmark("Convex Hull",
                           lambda pts: train_convex_hull(pts),
                           lambda pts, A, b, hull: classify_convex_hull(pts, A, b),
                           points)

    mean, cov_inv, k = benchmark("Ellipsoid (Mahalanobis)",
                                 lambda pts: train_ellipsoid(pts, percentile=0.99),
                                 lambda pts, mean, cov_inv, k: classify_ellipsoid(pts, mean, cov_inv, k),
                                 points)

    pca, mins, maxs = benchmark("PCA Oriented Bounding Box",
                                lambda pts: train_pca_obb(pts),
                                lambda pts, pca, mins, maxs: classify_pca_obb(pts, pca, mins, maxs),
                                points)

    gmm, gmm_threshold = benchmark("Gaussian Mixture Model",
                                   lambda pts: train_gmm(pts, n_components=3),
                                   lambda pts, gmm, threshold: classify_gmm(pts, gmm, threshold),
                                   points)

    (ocsvm_model,) = benchmark("One-Class SVM",
                               lambda pts: train_ocsvm(pts, nu=0.01),
                               lambda pts, model: classify_ocsvm(pts, model),
                               points)

    # ============================================================
    # Separate Plots (Option B)
    # ============================================================

    # 1. Convex Hull
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_point_cloud(ax, points)
    plot_convex_hull(ax, points, hull)
    ax.set_title("Convex Hull Region")
    plt.show()

    # 2. Ellipsoid
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_point_cloud(ax, points)
    ellipsoid_pts = sample_ellipsoid(mean, cov_inv, k)
    plot_ellipsoid(ax, ellipsoid_pts)
    ax.set_title("Ellipsoid Region")
    plt.show()

    # 3. PCA OBB
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_point_cloud(ax, points)
    obb_pts = obb_corners(pca, mins, maxs)
    plot_obb(ax, obb_pts)
    ax.set_title("PCA Oriented Bounding Box")
    plt.show()

    # 4. GMM Region
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_point_cloud(ax, points)
    plot_gmm_region(ax, gmm, gmm_threshold)
    ax.set_title("GMM Region")
    plt.show()

    # 5. OC-SVM Region
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_point_cloud(ax, points)
    plot_ocsvm_region(ax, ocsvm_model)
    ax.set_title("One-Class SVM Region")
    plt.show()
