from __future__ import annotations

import argparse
import json
from typing import List
import sys

import numpy as np
from PIL import Image

from quickdraw_fetcher import fetch_categories, fetch_random_image_and_label
from processor import to_binary_matrix, preprocess_to_28x28
from emoji_fetcher import (
    fetch_emoji_catalog,
    fetch_random_emoji_image,
    fetch_random_from_custom_urls,
)


def matrix_to_lists(matrix: np.ndarray) -> List[List[int]]:
    return [[int(v) for v in row] for row in matrix.tolist()]


def render_matrix(matrix: np.ndarray, style: str = "blocks") -> str:
    """Return a human-friendly string representation of a binary matrix.

    styles:
      - blocks: use full block for 1 and middle dot for 0 (best visual density)
      - ascii:  use # for 1 and . for 0
      - numbers: use 1 and 0
    """

    if style == "blocks":
        on, off = "█", "·"
    elif style == "ascii":
        on, off = "#", "."
    elif style == "numbers":
        on, off = "1", "0"
    else:
        raise ValueError("Unknown style; choose from: blocks, ascii, numbers")

    lines = []
    for row in matrix:
        line = " ".join(on if int(v) == 1 else off for v in row)
        lines.append(line)
    return "\n".join(lines)


def run_cli() -> None:
    def die(message: str) -> None:
        print(message, file=sys.stderr)
        raise SystemExit(1)
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a random labeled doodle image (QuickDraw), convert to 7x7 binary matrix, and print as JSON."
        )
    )
    parser.add_argument(
        "--source",
        choices=["emoji", "quickdraw"],
        default="emoji",
        help="Data source: emoji (default) or quickdraw",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Threshold 0-255; pixels >= threshold are treated as black (1)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert intensities before thresholding (use if output seems inverted)",
    )
    parser.add_argument(
        "--method",
        choices=["avg", "max", "fraction"],
        default="avg",
        help="Downsampling method: avg (area avg), max (any stroke), fraction (require portion)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="Portion of ON pixels per cell for method=fraction (0..1)",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["global", "otsu"],
        default="global",
        help="Thresholding mode: global fixed threshold or Otsu auto-threshold",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Also print a human-readable 7x7 matrix",
    )
    parser.add_argument(
        "--style",
        choices=["blocks", "ascii", "numbers"],
        default="blocks",
        help="Visualization style for the 7x7 matrix",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="7x7",
        help="Output size as WIDTHxHEIGHT (e.g., 7x7, 10x12)",
    )
    parser.add_argument(
        "--save-original",
        type=str,
        help="Path to save the original 28x28 grayscale image as PNG",
    )
    parser.add_argument(
        "--save-raw",
        type=str,
        help="Path to save the raw source image (emoji PNG) as-is",
    )
    parser.add_argument(
        "--emoji-set",
        choices=["twemoji", "noto"],
        default="twemoji",
        help="Emoji image set when --source=emoji",
    )
    parser.add_argument(
        "--custom-emoji-url",
        action="append",
        help="Custom emoji image URL (can repeat). If provided, picks randomly from these.",
    )
    parser.add_argument(
        "--emoji-catalog-url",
        action="append",
        help="Override emoji catalog URL(s). Can repeat; first that works is used.",
    )
    parser.add_argument(
        "--autocontrast",
        action="store_true",
        help="Apply autocontrast before resizing/binarization (recommended for emojis)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction (>1 darkens midtones, <1 brightens)",
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help="Gaussian blur radius before processing (reduces stripe artifacts)",
    )
    parser.add_argument(
        "--emoji-fetch-retries",
        type=int,
        default=5,
        help="Number of retry attempts when fetching a random emoji image fails",
    )
    parser.add_argument(
        "--label-mode",
        choices=["alias", "emoji"],
        default="alias",
        help="Label format: alias (text) or the emoji character",
    )
    args = parser.parse_args()

    # Validate and parse size
    try:
        width_str, height_str = args.size.lower().split("x", 1)
        out_w, out_h = int(width_str), int(height_str)
        if out_w <= 0 or out_h <= 0:
            raise ValueError
    except Exception:
        die("--size must be in format WIDTHxHEIGHT with positive integers, e.g., 7x7")

    # Validate threshold/fraction
    if not (0 <= args.threshold <= 255):
        die("--threshold must be in range 0..255")
    if args.method == "fraction" and not (0.0 <= args.fraction <= 1.0):
        die("--fraction must be in range 0..1 for method=fraction")

    # Fetch and prepare source image
    try:
        emoji_char = None
        if args.source == "quickdraw":
            try:
                categories = fetch_categories()
            except Exception as e:
                die(f"Failed to fetch QuickDraw categories: {e}")
            if not categories:
                die("QuickDraw categories list is empty")
            try:
                label, img28 = fetch_random_image_and_label(categories=categories)
            except Exception as e:
                die(f"Failed to fetch QuickDraw image: {e}")
            if not isinstance(img28, np.ndarray) or img28.shape != (28, 28):
                die("QuickDraw image has unexpected shape; expected 28x28 array")
            raw_img = Image.fromarray(img28.astype(np.uint8), mode="L")
        else:
            if args.custom_emoji_url:
                urls = [u for u in args.custom_emoji_url if (u and u.strip())]
                if not urls:
                    die("--custom-emoji-url provided but no valid URLs found")
                try:
                    label, raw_img, _ = fetch_random_from_custom_urls(urls)
                except Exception as e:
                    die(f"Failed to fetch custom emoji image: {e}")
            else:
                catalog = fetch_emoji_catalog(urls=args.emoji_catalog_url)
                if not catalog:
                    die("Emoji catalog is empty; cannot select a random emoji")
                # Retry fetching random emoji image a few times to avoid 404s
                last_err = None
                for _ in range(max(1, args.emoji_fetch_retries)):
                    try:
                        label, raw_img, _ = fetch_random_emoji_image(catalog=catalog, emoji_set=args.emoji_set)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        continue
                if last_err is not None:
                    die(f"Failed to fetch emoji image after {args.emoji_fetch_retries} attempts: {last_err}")
                emoji_char = label
                # Optionally replace label with alias text
                if args.label_mode == "alias":
                    try:
                        match = next((it for it in catalog if it.emoji == label), None)
                        if match:
                            if match.aliases:
                                label = match.aliases[0].replace("_", " ")
                            else:
                                label = (match.description or "emoji").replace("_", " ")
                    except Exception:
                        pass
            # Convert raw emoji to 28x28 grayscale with preprocessing
            try:
                img28 = preprocess_to_28x28(
                    raw_img,
                    autocontrast=bool(args.autocontrast),
                    gamma=float(args.gamma),
                    blur_radius=float(args.blur),
                )
            except Exception as e:
                die(f"Failed to convert emoji image to 28x28 grayscale: {e}")
            if img28.shape != (28, 28):
                die("Converted emoji image has unexpected shape; expected 28x28")
    except SystemExit:
        raise
    except Exception as e:
        die(f"Unexpected error during fetching/conversion: {e}")
    try:
        width_str, height_str = args.size.lower().split("x", 1)
        out_w, out_h = int(width_str), int(height_str)
        if out_w <= 0 or out_h <= 0:
            raise ValueError
    except Exception:
        raise SystemExit("--size must be in format WIDTHxHEIGHT, e.g., 7x7")

    # Auto-polarity: center color becomes foreground (black squares), borders become background
    # Compute on the preprocessed 28x28 grayscale image
    c0, c1 = 10, 18  # central region ~8x8
    center_mean = float(np.mean(img28[c0:c1, c0:c1]))
    border_pixels = np.concatenate([
        img28[0:4, :].flatten(), img28[-4:, :].flatten(), img28[:, 0:4].flatten(), img28[:, -4:].flatten()
    ])
    border_mean = float(np.mean(border_pixels)) if border_pixels.size else center_mean
    auto_invert = center_mean < border_mean

    try:
        matrix = to_binary_matrix(
            img28,
            out_width=out_w,
            out_height=out_h,
            threshold=args.threshold,
            invert=(args.invert or auto_invert),
            method=args.method,
            fraction_required=args.fraction,
            threshold_mode=args.threshold_mode,
        )
    except Exception as e:
        die(f"Failed to create binary matrix: {e}")

    output = {
        "label": label,
        "matrix": matrix_to_lists(matrix),
    }
    try:
        print(json.dumps(output, ensure_ascii=False))
    except Exception as e:
        die(f"Failed to print JSON output: {e}")

    if args.print_matrix:
        print("\nlabel:", label)
        if args.source == "emoji" and emoji_char:
            print("emoji:", emoji_char)
        print(f"{out_w}x{out_h}:")
        print(render_matrix(matrix, style=args.style))

    if args.save_original:
        try:
            img = Image.fromarray(img28.astype(np.uint8), mode="L")
            img.save(args.save_original)
        except Exception as e:
            die(f"Failed to save --save-original image: {e}")
    if args.save_raw and raw_img is not None:
        try:
            raw_img.save(args.save_raw)
        except Exception as e:
            die(f"Failed to save --save-raw image: {e}")


if __name__ == "__main__":
    run_cli()


