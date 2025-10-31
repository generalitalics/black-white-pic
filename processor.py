from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps, ImageFilter


def to_binary_matrix(
    image_28x28: np.ndarray,
    out_width: int = 7,
    out_height: int = 7,
    threshold: int = 128,
    invert: bool | None = None,
    method: str = "avg",
    fraction_required: float = 0.2,
    threshold_mode: str = "global",
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

    # Compute source threshold if needed
    thr_src = threshold
    if threshold_mode == "otsu":
        thr_src = _otsu_threshold(arr)

    if method == "avg":
        # Area averaging then global threshold
        pil_img = Image.fromarray(arr, mode="L")
        pil_small = pil_img.resize((out_width, out_height), resample=Image.BOX)
        small = np.asarray(pil_small, dtype=np.uint8)
        thr = threshold if threshold_mode == "global" else _otsu_threshold(small)
        binary = (small >= thr).astype(np.uint8)
        return binary

    # For methods based on presence within the block, first threshold at source resolution
    highres_bin = (arr >= thr_src).astype(np.uint8)  # 0/1
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


def _otsu_threshold(arr: np.ndarray) -> int:
    hist = np.bincount(arr.flatten(), minlength=256).astype(np.float64)
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return int(threshold)


def preprocess_to_28x28(
    image: Image.Image,
    autocontrast: bool = True,
    gamma: float = 1.0,
    blur_radius: float = 0.0,
) -> np.ndarray:
    # Composite on white to handle transparency, then compute luminance
    img = image.convert("RGBA")
    r, g, b, a = [np.asarray(ch, dtype=np.float32) for ch in img.split()]
    alpha = a / 255.0
    # composite over white
    r_c = r * alpha + 255.0 * (1.0 - alpha)
    g_c = g * alpha + 255.0 * (1.0 - alpha)
    b_c = b * alpha + 255.0 * (1.0 - alpha)
    # Rec.709 luminance
    y = 0.2126 * r_c + 0.7152 * g_c + 0.0722 * b_c
    y = np.clip(y, 0, 255).astype(np.uint8)
    pil_y = Image.fromarray(y, mode="L")
    if blur_radius and blur_radius > 0:
        pil_y = pil_y.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    if autocontrast:
        pil_y = ImageOps.autocontrast(pil_y)
    if gamma and gamma != 1.0:
        arr = np.asarray(pil_y, dtype=np.float32) / 255.0
        arr = np.power(arr, 1.0 / float(gamma))
        pil_y = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    pil_28 = pil_y.resize((28, 28), resample=Image.BOX)
    return np.asarray(pil_28, dtype=np.uint8)


