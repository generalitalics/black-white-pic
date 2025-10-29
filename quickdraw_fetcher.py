from __future__ import annotations

import io
import random
import urllib.parse
from typing import List, Tuple

import numpy as np
import requests


CATEGORIES_TXT_URL = (
    "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
)
NUMPY_BITMAP_BASE_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{label}.npy"
)


def fetch_categories(timeout_seconds: float = 10.0) -> List[str]:
    response = requests.get(CATEGORIES_TXT_URL, timeout=timeout_seconds)
    response.raise_for_status()
    # Each line is a label; keep as-is (they are lowercase, can include spaces)
    categories = [line.strip() for line in response.text.splitlines() if line.strip()]
    return categories


def fetch_random_image_and_label(
    categories: List[str] | None = None,
    timeout_seconds: float = 20.0,
    rng: random.Random | None = None,
) -> Tuple[str, np.ndarray]:
    """
    Fetch a random labeled bitmap from Google's QuickDraw public dataset.

    Returns (label, image) where image is a (28, 28) uint8 numpy array.
    """
    if rng is None:
        rng = random

    if not categories:
        categories = fetch_categories(timeout_seconds=timeout_seconds)
    label = rng.choice(categories)

    label_encoded = urllib.parse.quote(label)
    url = NUMPY_BITMAP_BASE_URL.format(label=label_encoded)

    # The file is an npy of shape (N, 784) uint8, each row is a flattened 28x28 image
    resp = requests.get(url, timeout=timeout_seconds)
    resp.raise_for_status()

    with io.BytesIO(resp.content) as f:
        data = np.load(f)

    if data.ndim != 2 or data.shape[1] != 28 * 28:
        raise ValueError(f"Unexpected numpy bitmap shape: {data.shape}")

    idx = rng.randrange(0, data.shape[0])
    flat = data[idx]
    img = flat.reshape(28, 28).astype(np.uint8)
    return label, img


