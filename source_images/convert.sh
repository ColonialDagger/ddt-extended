#!/usr/bin/env bash
set -euo pipefail

if ! command -v identify >/dev/null 2>&1; then
    echo "Error: 'identify' (ImageMagick) is required but not installed."
    echo "Run:"
    echo "    sudo apt install imagemagick"
    echo "to install."
    exit 1
fi

for file in *.png; do
    # Skip if no PNGs exist
    [[ -e "$file" ]] || { echo "No PNG files found."; exit 0; }

    # Extract width and height
    dimensions=$(identify -format "%w %h" "$file")
    width=$(echo "$dimensions" | awk '{print $1}')
    height=$(echo "$dimensions" | awk '{print $2}')

    new_name="${width}x${height}p.png"

    # Avoid overwriting an existing file
    if [[ -e "$new_name" ]]; then
        echo "Skipping '$file' -> '$new_name' already exists."
        continue
    fi

    echo "Renaming '$file' -> '$new_name'"
    mv -- "$file" "$new_name"
done

