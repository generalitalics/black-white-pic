from __future__ import annotations

import io
import random
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from PIL import Image


# Multiple online catalogs (tried in order until one works/non-empty)
GEMOJI_JSON_URLS_DEFAULT = [
    # GitHub raw
    "https://raw.githubusercontent.com/github/gemoji/master/db/emoji.json",
    # jsDelivr mirror of the repository
    "https://cdn.jsdelivr.net/gh/github/gemoji@master/db/emoji.json",
    # FastGit mirror
    "https://raw.fastgit.org/github/gemoji/master/db/emoji.json",
    # iamcal/emoji-data (widely used; contains unified, short_names, etc.)
    "https://raw.githubusercontent.com/iamcal/emoji-data/master/emoji.json",
    "https://cdn.jsdelivr.net/gh/iamcal/emoji-data@master/emoji.json",
]


@dataclass
class EmojiItem:
    emoji: str
    description: str
    unified: str  # e.g., "1F600" or sequences like "1F469-200D-1F4BB"
    aliases: List[str]


def fetch_emoji_catalog(
    timeout_seconds: float = 15.0,
    urls: Optional[List[str]] = None,
) -> List[EmojiItem]:
    sources = urls or GEMOJI_JSON_URLS_DEFAULT
    for url in sources:
        try:
            resp = requests.get(url, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            items: List[EmojiItem] = []
            for e in data:
                # Detect schema: gemoji vs iamcal
                if e.get("unified"):
                    unified = str(e["unified"]).lower()
                else:
                    continue

                # Try gemoji (has 'emoji' char and 'aliases')
                if "emoji" in e and ("aliases" in e or "description" in e or "name" in e):
                    emoji_char = e.get("emoji") or _unified_to_emoji_str(unified)
                    description = e.get("description") or e.get("name") or "emoji"
                    aliases = list(e.get("aliases") or [])
                    items.append(EmojiItem(emoji=emoji_char, description=description, unified=unified, aliases=aliases))
                    continue

                # Try iamcal schema (has 'short_name'/'short_names')
                if "short_names" in e or "short_name" in e:
                    aliases = list(e.get("short_names") or ([] if not e.get("short_name") else [e.get("short_name")]))
                    description = e.get("name") or (aliases[0] if aliases else "emoji")
                    emoji_char = _unified_to_emoji_str(unified)
                    items.append(EmojiItem(emoji=emoji_char, description=description, unified=unified, aliases=aliases))
                    continue
            if items:
                return items
        except Exception:
            continue
    return []


def _unified_to_emoji_str(unified: str) -> str:
    """Convert a unified code like '1f469-200d-1f4bb' into a Unicode string."""
    try:
        parts = unified.split("-")
        codepoints = [int(p, 16) for p in parts if p]
        return "".join(chr(cp) for cp in codepoints)
    except Exception:
        return ""



def unified_to_twemoji_png_url(unified: str, size: int = 72) -> str:
    # Twemoji files are hyphen-joined lowercase hex, e.g. 1f600.png, 1f469-200d-1f4bb.png
    # CDN path for PNG sizes: 72x72
    # Source: https://github.com/twitter/twemoji
    return (
        f"https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/{size}x{size}/{unified}.png"
    )


def unified_to_noto_png_url(unified: str, size: int = 128) -> str:
    # Noto Emoji PNG naming: emoji_uXXXX[_XXXX].png, lowercase hex.
    # Source repo: https://github.com/googlefonts/noto-emoji
    parts = unified.split("-")
    joined = "_".join(parts)
    return (
        f"https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/{size}/emoji_u{joined}.png"
    )


def fetch_image_from_url(url: str, timeout_seconds: float = 20.0) -> Image.Image:
    r = requests.get(url, timeout=timeout_seconds)
    r.raise_for_status()
    with io.BytesIO(r.content) as bio:
        img = Image.open(bio)
        img.load()
    return img


def fetch_random_emoji_image(
    catalog: Optional[List[EmojiItem]] = None,
    rng: Optional[random.Random] = None,
    emoji_set: str = "twemoji",
) -> Tuple[str, Image.Image, str]:
    """
    Returns (label, PIL.Image, source_url).
    label is the emoji character itself.
    """
    if rng is None:
        rng = random
    if not catalog:
        catalog = fetch_emoji_catalog()
    item = rng.choice(catalog)

    if emoji_set == "twemoji":
        url = unified_to_twemoji_png_url(item.unified)
    elif emoji_set == "noto":
        url = unified_to_noto_png_url(item.unified)
    else:
        raise ValueError("emoji_set must be 'twemoji' or 'noto'")

    img = fetch_image_from_url(url)
    return item.emoji, img, url


def fetch_random_from_custom_urls(
    urls: List[str], rng: Optional[random.Random] = None
) -> Tuple[str, Image.Image, str]:
    if not urls:
        raise ValueError("No custom emoji URLs provided")
    if rng is None:
        rng = random
    url = rng.choice(urls)
    img = fetch_image_from_url(url)
    # Use the URL basename as label fallback
    label = url.split("/")[-1] or "custom"
    return label, img, url


