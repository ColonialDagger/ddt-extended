import csv
import numpy as np
import matplotlib.pyplot as plt

def load_rgb_csv(path):
    """
    Loads a CSV of rows: r,g,b
    Returns three NumPy arrays: R, G, B
    """
    r_vals = []
    g_vals = []
    b_vals = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue  # skip malformed rows
            try:
                r, g, b = map(int, row)
                r_vals.append(r)
                g_vals.append(g)
                b_vals.append(b)
            except ValueError:
                # skip header or bad rows
                continue

    return np.array(r_vals), np.array(g_vals), np.array(b_vals)


def plot_rgb_3d(r, g, b):
    """
    Creates a 3D scatterplot of RGB values.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize colors for plotting (0–1 range)
    colors = np.stack([r, g, b], axis=1) / 255.0

    ax.scatter(r, g, b, c=colors, marker="o", s=8, alpha=0.8)

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    ax.set_title("3D RGB Scatterplot")

    plt.tight_layout()
    plt.show()


def main():
    csv_path = "pixels_unique.csv"  # change if needed

    r, g, b = load_rgb_csv(csv_path)
    print(f"Loaded {len(r)} pixels from {csv_path}")

    plot_rgb_3d(r, g, b)


if __name__ == "__main__":
    main()
