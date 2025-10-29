from __future__ import annotations

import numpy as np
from PIL import Image


def to_binary_matrix(
    image_28x28: np.ndarray,
    out_width: int = 7,
    out_height: int = 7,
    threshold: int = 128,
    invert: bool | None = None,
    method: str = "avg",
    fraction_required: float = 0.2,
) -> np.ndarray:
    """
    Convert a 28x28 grayscale image (uint8 0..255) to an out_height x out_width binary matrix of 0/1.

    1 means a black/inked pixel; 0 means empty/white.

    QuickDraw bitmaps typically have strokes as high values on a white background.
    If you find the output inverted for your use case, set invert=True.
    """
    if image_28x28.shape != (28, 28):
        raise ValueError("Input image must be 28x28")

    arr = image_28x28.astype(np.uint8)

    if invert is True:
        arr = 255 - arr

    if method == "avg":
        # Area averaging then global threshold
        pil_img = Image.fromarray(arr, mode="L")
        pil_small = pil_img.resize((out_width, out_height), resample=Image.BOX)
        small = np.asarray(pil_small, dtype=np.uint8)
        binary = (small >= threshold).astype(np.uint8)
        return binary

    # For methods based on presence within the block, first threshold at source resolution
    highres_bin = (arr >= threshold).astype(np.uint8)  # 0/1
    pil_bin = Image.fromarray((highres_bin * 255).astype(np.uint8), mode="L")
    pil_bin_small = pil_bin.resize((out_width, out_height), resample=Image.BOX)
    # After BOX on binary [0,255], values are proportional to fraction of ON pixels in each block
    frac = np.asarray(pil_bin_small, dtype=np.float32) / 255.0  # in [0,1]

    if method == "max":
        # Any stroke presence in the block turns the cell ON
        return (frac > 0.0).astype(np.uint8)
    elif method == "fraction":
        # Require at least a certain portion of pixels ON in the block
        return (frac >= float(fraction_required)).astype(np.uint8)
    else:
        raise ValueError("Unknown method; choose from: avg, max, fraction")


