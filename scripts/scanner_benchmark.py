import numpy as np
import time

def make_bar(width, height, fraction, flicker=False, noise=0.0):
    """
    Create a synthetic health bar:
    - fraction: 0.0 → 1.0
    - flicker: randomly remove columns (Final Stand simulation)
    - noise: random pixel noise in the bar
    """
    bar = np.zeros((height, width), dtype=np.uint8)

    # Fill portion
    fill_w = int(width * fraction)
    bar[:, :fill_w] = 1

    # Final Stand flicker simulation
    if flicker and fill_w > 0:
        gaps = np.random.choice(fill_w, size=max(1, fill_w // 20), replace=False)
        bar[:, gaps] = 0

    # Noise injection
    if noise > 0:
        noise_mask = np.random.rand(height, width) < noise
        bar = np.bitwise_xor(bar, noise_mask.astype(np.uint8))

    return bar


def sum_estimator(healthy_mask, neg_mask):
    total = np.count_nonzero(neg_mask)
    if total == 0:
        return 0.0
    return np.count_nonzero(healthy_mask) / total


def scan_estimator(healthy_mask, neg_mask, min_col_fraction=0.6):
    h, w = healthy_mask.shape
    mc = np.count_nonzero(neg_mask, axis=0)
    hc = np.count_nonzero(healthy_mask, axis=0)
    thr = mc * min_col_fraction

    # Find first full column from right
    first_full = -1
    for col in range(w - 1, -1, -1):
        if mc[col] > 0 and hc[col] >= thr[col]:
            first_full = col
            break

    if first_full < 0:
        return 0.0

    # Everything left of first_full is assumed full
    full_cols = mc[:first_full].sum()
    edge_cols = hc[first_full:first_full+1].sum()

    total = mc.sum()
    return (full_cols + edge_cols) / total


def run_benchmark():
    width = 300
    height = 20

    scenarios = [
        ("100% full", 1.00, False, 0.0),
        ("75% health", 0.75, False, 0.0),
        ("50% health", 0.50, False, 0.0),
        ("25% health", 0.25, False, 0.0),
        ("10% health", 0.10, False, 0.0),
        ("Final Stand flicker", 0.10, True, 0.0),
        ("Noisy mask", 0.75, False, 0.05),
    ]

    for name, frac, flicker, noise in scenarios:
        bar = make_bar(width, height, frac, flicker=flicker, noise=noise)

        # In synthetic tests, neg_mask = all ones
        neg_mask = np.ones_like(bar)
        healthy_mask = bar.copy()

        # Timing
        t0 = time.perf_counter()
        s1 = sum_estimator(healthy_mask, neg_mask)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        s2 = scan_estimator(healthy_mask, neg_mask)
        t3 = time.perf_counter()

        print(f"\n{name}")
        print(f"  True fraction: {frac:.3f}")
        print(f"  Sum estimator: {s1:.3f}   ({(t1 - t0)*1e6:.1f} µs)")
        print(f"  Scan estimator: {s2:.3f}  ({(t3 - t2)*1e6:.1f} µs)")

if __name__ == "__main__":
    run_benchmark()