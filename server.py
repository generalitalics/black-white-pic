from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from db_config import get_db_params
from emoji_fetcher import (
    fetch_emoji_catalog,
    fetch_random_emoji_image,
    fetch_random_from_custom_urls,
)
from processor import to_binary_matrix, preprocess_to_28x28


class LoginRequest(BaseModel):
    username: str
    password: str


app = FastAPI(title="Nonogram Service")


def matrix_to_lists(matrix: np.ndarray) -> List[List[int]]:
    return [[int(v) for v in row] for row in matrix.tolist()]


@app.get("/nonogram/create")
def create_nonogram(
    size: str = Query("10x10", description="Output size as WIDTHxHEIGHT, e.g., 10x10"),
    emoji_set: str = Query("twemoji", pattern="^(twemoji|noto)$"),
    label_mode: str = Query("alias", pattern="^(alias|emoji)$"),
    method: str = Query("fraction", pattern="^(avg|max|fraction)$"),
    fraction: float = Query(0.25, ge=0.0, le=1.0),
    threshold: int = Query(128, ge=0, le=255),
    threshold_mode: str = Query("otsu", pattern="^(global|otsu)$"),
    autocontrast: bool = True,
    gamma: float = 1.0,
    blur: float = 0.4,
    emoji_fetch_retries: int = 5,
    emoji_catalog_url: Optional[List[str]] = Query(None),
    custom_emoji_url: Optional[List[str]] = Query(None),
):
    # Parse size
    try:
        w_str, h_str = size.lower().split("x", 1)
        out_w, out_h = int(w_str), int(h_str)
        if out_w <= 0 or out_h <= 0:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT with positive integers")

    # Fetch emoji image (online only)
    try:
        if custom_emoji_url:
            urls = [u for u in custom_emoji_url if u and u.strip()]
            if not urls:
                raise HTTPException(status_code=400, detail="custom_emoji_url is empty")
            label, raw_img, _ = fetch_random_from_custom_urls(urls)
        else:
            catalog = fetch_emoji_catalog(urls=emoji_catalog_url)
            if not catalog:
                raise HTTPException(status_code=503, detail="Emoji catalog is empty/unavailable")
            last_err: Optional[Exception] = None
            for _ in range(max(1, int(emoji_fetch_retries))):
                try:
                    label, raw_img, _ = fetch_random_emoji_image(catalog=catalog, emoji_set=emoji_set)
                    last_err = None
                    break
                except Exception as e:  # e.g., 404 for some composed sequences
                    last_err = e
                    continue
            if last_err is not None:
                raise HTTPException(status_code=502, detail=f"Failed to fetch emoji image: {last_err}")

        emoji_char = label
        # Convert label to alias text if requested
        if label_mode == "alias" and not custom_emoji_url:
            try:
                match = next((it for it in catalog if it.emoji == label), None)
                if match:
                    if match.aliases:
                        label = match.aliases[0].replace("_", " ")
                    else:
                        label = (match.description or "emoji").replace("_", " ")
            except Exception:
                pass

        # Preprocess to 28x28 grayscale
        img28 = preprocess_to_28x28(
            raw_img,
            autocontrast=bool(autocontrast),
            gamma=float(gamma),
            blur_radius=float(blur),
        )
        if img28.shape != (28, 28):
            raise HTTPException(status_code=500, detail="Unexpected preprocessing shape")

        # Auto polarity based on center vs borders
        c0, c1 = 10, 18
        center_mean = float(np.mean(img28[c0:c1, c0:c1]))
        border_pixels = np.concatenate([
            img28[0:4, :].flatten(), img28[-4:, :].flatten(), img28[:, 0:4].flatten(), img28[:, -4:].flatten()
        ])
        border_mean = float(np.mean(border_pixels)) if border_pixels.size else center_mean
        auto_invert = center_mean < border_mean

        matrix = to_binary_matrix(
            img28,
            out_width=out_w,
            out_height=out_h,
            threshold=int(threshold),
            invert=bool(auto_invert),
            method=method,
            fraction_required=float(fraction),
            threshold_mode=threshold_mode,
        )

        return {
            "emoji": emoji_char,
            "label": label,
            "matrix": matrix_to_lists(matrix),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/login")
def login(credentials: LoginRequest):
    """
    Authenticate user by checking username and password in the database.
    Returns user info if credentials are valid, raises HTTPException otherwise.
    """
    conn = None
    try:
        # Connect to database
        db_params = get_db_params()
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Check user credentials
        cursor.execute(
            "SELECT id, username, is_admin FROM users WHERE username = %s AND password = %s",
            (credentials.username, credentials.password),
        )
        result = cursor.fetchone()

        if result is None:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        user_id, username, is_admin = result

        return {
            "success": True,
            "user": {
                "id": user_id,
                "username": username,
                "is_admin": is_admin,
            },
        }
    except HTTPException:
        raise
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn:
            conn.close()


# For local debug: uvicorn server:app --reload

