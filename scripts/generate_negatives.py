import os
import pathlib

from utils import generate_negative_file_from_image


def list_pngs(dir):
    dir = pathlib.Path(dir)
    if not dir.is_dir():
        raise NotADirectoryError(f"{dir} is not a directory")
    return [str(p) for p in dir.iterdir() if p.is_file() & p.name.endswith(".png")]

def main():
    lookup_table = {
        "1280x720": 652,
        "1280x800": 692,
        "1280x1024": 803,
        "1600x1200": 964,
        "1680x720": 652,
        "1680x1050": 908,
        "1720x1440": 1112,
        "1920x820": 741,
        "1920x1080": 977,
        "1920x1200": 1038,
        "2100x990": 896,
        "2256x1504": 1267,
        "2310x990": 896,
        "2560x1080": 977,
        "2560x1440": 1305,
        "3024x1296": 1173,
        "3440x1440": 1303,
        "3840x2160": 1954,
    }

    files = list_pngs("source_images")
    for f in files:
        print(f"Processing {f}...", end="")
        generate_negative_file_from_image(
            image_path=f,
            seed_y=lookup_table[f.split("\\")[1][:-5]],
            tolerance=0.20,
            output_dir="negatives",
        )
        print(" Done!")

if __name__ == "__main__":
    main()