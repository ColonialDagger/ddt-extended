import ctypes
import cv2
import os

import numpy as np

@staticmethod
def generate_negative_file_from_image(
        image_path: str,
        seed_y: int,
        tolerance: float = 0.20,
        output_dir: str = "negatives"
) -> np.ndarray:
    """
    Creates a full-resolution binary mask using a local, contiguous region-grow based on the color
    similarity of the seed pixel. This is like using the magic wand tool in paint.net.

    :param image_path: Path to the image.
    :param seed_y: Y coordinate of a pixel on the healthbar. X is automatically chosen as the center of the screen.
    :param tolerance: Fractional tolerance
    :param output_dir: Directory where the mask PNG will be saved.
    :return: np.ndarray (H,W,4) uint8
    """

    # Load image (BGR)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w, _ = img.shape
    seed_x = w // 2

    if not (0 <= seed_y < h):
        raise ValueError("Seed Y coordinate is outside image bounds")

    # Seed pixel color
    seed_color = img[seed_y, seed_x].astype(np.int32)

    # Precompute max RGB distance
    max_dist = np.sqrt(3 * (255 ** 2))
    threshold = tolerance * max_dist

    # Output mask (boolean)
    visited = np.zeros((h, w), dtype=bool)
    mask_bool = np.zeros((h, w), dtype=bool)

    # BFS queue
    from collections import deque
    q = deque()
    q.append((seed_x, seed_y))
    visited[seed_y, seed_x] = True

    # 4-neighborhood
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        x, y = q.popleft()

        # Check similarity
        diff = img[y, x].astype(np.int32) - seed_color
        dist = np.sqrt(np.sum(diff * diff))

        if dist <= threshold:
            mask_bool[y, x] = True

            # Explore neighbors
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((nx, ny))

    # Convert boolean mask → RGBA
    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[mask_bool] = [255, 255, 255, 255]  # white, opaque
    mask_rgba[~mask_bool] = [0, 0, 0, 0]  # transparent

    # Determine resolution string
    resolution = f"{w}x{h}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save PNG
    out_path = os.path.join(output_dir, f"{resolution}_negative.png")
    cv2.imwrite(out_path, mask_rgba)

    return mask_rgba

@staticmethod
def is_window_open_win32(window_title: str) -> bool:
    """
    Checks if the given window title is open and visible.

    :param window_title: Name of the window to check.
    :return:
    """
    # Define the necessary Win32 API functions
    user32 = ctypes.windll.user32
    # FindWindow returns an HWND (window handle) if found, otherwise NULL
    hwnd = user32.FindWindowW(None, window_title)
    if hwnd:
        # Check if the window is visible (optional, a window handle might exist but be hidden)
        is_visible = user32.IsWindowVisible(hwnd)
        if is_visible:
            return True
    return False