from __future__ import annotations

import os
from typing import List, Optional

import json
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


class ProgressRequest(BaseModel):
    username: str
    level: int  # Sequential level number within difficulty (1, 2, 3...)
    difficulty: str  # 'easy', 'medium', 'hard'
    matrix: List[List[int]]  # 2D array of 0s and 1s
    reason: str  # 'manual', 'back', 'next_level', 'completed'


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


@app.post("/progress")
def save_progress(progress: ProgressRequest):
    """
    Save or update user progress for a level.
    Accepts: username, level (number), difficulty, matrix, reason
    Uses UPSERT (INSERT ... ON CONFLICT UPDATE) to handle existing records.
    """
    conn = None
    try:
        # Validate reason
        valid_reasons = ["manual", "back", "next_level", "completed"]
        if progress.reason not in valid_reasons:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid reason. Must be one of: {', '.join(valid_reasons)}",
            )

        # Validate difficulty
        valid_difficulties = ["easy", "medium", "hard"]
        difficulty_lower = progress.difficulty.lower()
        if difficulty_lower not in valid_difficulties:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid difficulty. Must be one of: {', '.join(valid_difficulties)}",
            )

        # Map reason to status
        status = "completed" if progress.reason == "completed" else "in_progress"

        # Connect to database
        db_params = get_db_params()
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Find user_id by username
        cursor.execute("SELECT id FROM users WHERE username = %s", (progress.username,))
        user_result = cursor.fetchone()
        if user_result is None:
            raise HTTPException(status_code=404, detail=f"User '{progress.username}' not found")
        user_id = user_result[0]

        # Find level_id by difficulty and level number
        cursor.execute(
            """
            SELECT l.id 
            FROM levels l
            JOIN difficulty d ON l.difficulty_id = d.id
            WHERE d.type = %s AND l.number = %s
            """,
            (difficulty_lower, progress.level),
        )
        level_result = cursor.fetchone()
        if level_result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Level {progress.level} with difficulty '{difficulty_lower}' not found",
            )
        level_id = level_result[0]

        # Prepare matrix as JSONB
        matrix_json = json.dumps(progress.matrix)

        # UPSERT: Insert or update on conflict
        cursor.execute(
            """
            INSERT INTO user_progress (user_id, level_id, status, reason, matrix, updated_at)
            VALUES (%s, %s, %s, %s, %s::jsonb, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, level_id)
            DO UPDATE SET
                status = EXCLUDED.status,
                reason = EXCLUDED.reason,
                matrix = EXCLUDED.matrix,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id, updated_at
            """,
            (user_id, level_id, status, progress.reason, matrix_json),
        )
        result = cursor.fetchone()
        conn.commit()

        progress_id, updated_at = result

        return {
            "success": True,
            "progress": {
                "id": progress_id,
                "username": progress.username,
                "level": progress.level,
                "difficulty": difficulty_lower,
                "status": status,
                "reason": progress.reason,
                "updated_at": updated_at.isoformat() if updated_at else None,
            },
        }
    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn:
            conn.close()


@app.post("/progress/save")
def save_progress_save(progress: ProgressRequest):
    """
    Alias for POST /progress - saves user progress.
    This endpoint exists to match frontend route /api/progress/save
    """
    return save_progress(progress)


@app.get("/progress/check")
def check_level_progress(
    username: str = Query(..., description="Username"),
    difficulty: str = Query(..., description="Difficulty: easy, medium, hard"),
    level: int = Query(..., description="Level number"),
):
    """
    Quick check if progress exists for a level (without loading matrix).
    Returns only metadata - useful for showing loader before full load.
    """
    conn = None
    try:
        # Validate difficulty
        valid_difficulties = ["easy", "medium", "hard"]
        difficulty_lower = difficulty.lower()
        if difficulty_lower not in valid_difficulties:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid difficulty. Must be one of: {', '.join(valid_difficulties)}",
            )

        db_params = get_db_params()
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Find user_id by username
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user_result = cursor.fetchone()
        if user_result is None:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        user_id = user_result[0]

        # Find level_id by difficulty and level number
        cursor.execute(
            """
            SELECT l.id 
            FROM levels l
            JOIN difficulty d ON l.difficulty_id = d.id
            WHERE d.type = %s AND l.number = %s
            """,
            (difficulty_lower, level),
        )
        level_result = cursor.fetchone()
        if level_result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Level {level} with difficulty '{difficulty_lower}' not found",
            )
        level_id = level_result[0]

        # Quick check - only get status and updated_at (no matrix)
        cursor.execute(
            """
            SELECT status, reason, updated_at
            FROM user_progress
            WHERE user_id = %s AND level_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id, level_id),
        )
        result = cursor.fetchone()

        if result is None:
            return {
                "success": True,
                "has_progress": False,
                "username": username,
                "level": level,
                "difficulty": difficulty_lower,
            }

        status, reason, updated_at = result
        return {
            "success": True,
            "has_progress": True,
            "username": username,
            "level": level,
            "difficulty": difficulty_lower,
            "status": status,
            "reason": reason,
            "updated_at": updated_at.isoformat() if updated_at else None,
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


@app.get("/progress/load")
def load_level_progress(
    username: str = Query(..., description="Username"),
    difficulty: str = Query(..., description="Difficulty: easy, medium, hard"),
    level: int = Query(..., description="Level number"),
):
    """
    Load level progress for a user. Returns the most recent saved matrix if exists.
    Used to restore game state when opening a level.
    """
    conn = None
    try:
        # Validate difficulty
        valid_difficulties = ["easy", "medium", "hard"]
        difficulty_lower = difficulty.lower()
        if difficulty_lower not in valid_difficulties:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid difficulty. Must be one of: {', '.join(valid_difficulties)}",
            )

        db_params = get_db_params()
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Find user_id by username
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user_result = cursor.fetchone()
        if user_result is None:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        user_id = user_result[0]

        # Find level_id by difficulty and level number
        cursor.execute(
            """
            SELECT l.id 
            FROM levels l
            JOIN difficulty d ON l.difficulty_id = d.id
            WHERE d.type = %s AND l.number = %s
            """,
            (difficulty_lower, level),
        )
        level_result = cursor.fetchone()
        if level_result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Level {level} with difficulty '{difficulty_lower}' not found",
            )
        level_id = level_result[0]

        # Get the most recent progress for this user and level (ordered by updated_at DESC)
        cursor.execute(
            """
            SELECT matrix, status, reason, updated_at
            FROM user_progress
            WHERE user_id = %s AND level_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id, level_id),
        )
        result = cursor.fetchone()

        if result is None:
            # No saved progress - return empty/null matrix
            return {
                "success": True,
                "has_progress": False,
                "username": username,
                "level": level,
                "difficulty": difficulty_lower,
                "matrix": None,
                "status": None,
                "reason": None,
            }

        matrix_json, status, reason, updated_at = result

        # Parse matrix if it exists
        matrix_data = None
        if matrix_json is not None:
            if isinstance(matrix_json, str):
                matrix_data = json.loads(matrix_json)
            else:
                matrix_data = matrix_json

        return {
            "success": True,
            "has_progress": True,
            "username": username,
            "level": level,
            "difficulty": difficulty_lower,
            "matrix": matrix_data,
            "status": status,
            "reason": reason,
            "updated_at": updated_at.isoformat() if updated_at else None,
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


@app.get("/progress")
def get_progress(
    username: Optional[str] = Query(None, description="Username (new format)"),
    difficulty: Optional[str] = Query(None, description="Difficulty: easy, medium, hard (new format)"),
    level: Optional[int] = Query(None, description="Level number (new format)"),
    user_id: Optional[int] = Query(None, description="User ID (legacy format)"),
    level_id: Optional[int] = Query(None, description="Level ID (legacy format)"),
):
    """
    Get user progress. Supports two formats:
    1. New: username + (optional) difficulty + level
    2. Legacy: user_id + (optional) level_id
    """
    conn = None
    try:
        db_params = get_db_params()
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # New format: username-based
        if username is not None:
            # Find user_id
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            user_result = cursor.fetchone()
            if user_result is None:
                raise HTTPException(status_code=404, detail=f"User '{username}' not found")
            user_id_db = user_result[0]

            if difficulty is not None and level is not None:
                # Get specific level progress
                difficulty_lower = difficulty.lower()
                cursor.execute(
                    """
                    SELECT up.id, up.user_id, up.level_id, up.status, up.reason, up.matrix, up.updated_at,
                           l.number, d.type
                    FROM user_progress up
                    JOIN levels l ON up.level_id = l.id
                    JOIN difficulty d ON l.difficulty_id = d.id
                    WHERE up.user_id = %s AND d.type = %s AND l.number = %s
                    """,
                    (user_id_db, difficulty_lower, level),
                )
                result = cursor.fetchone()
                if result is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Progress not found for user '{username}', difficulty '{difficulty}', level {level}",
                    )
                (
                    progress_id,
                    uid,
                    lid,
                    status,
                    reason,
                    matrix_json,
                    updated_at,
                    level_num,
                    diff_type,
                ) = result
                matrix_data = matrix_json
                if isinstance(matrix_json, str):
                    matrix_data = json.loads(matrix_json)
                return {
                    "success": True,
                    "progress": {
                        "id": progress_id,
                        "username": username,
                        "level": level_num,
                        "difficulty": diff_type,
                        "status": status,
                        "reason": reason,
                        "matrix": matrix_data,
                        "updated_at": updated_at.isoformat() if updated_at else None,
                    },
                }
            else:
                # Get all progress for user with difficulty and level info
                cursor.execute(
                    """
                    SELECT up.id, up.user_id, up.level_id, up.status, up.reason, up.matrix, up.updated_at,
                           l.number, d.type
                    FROM user_progress up
                    JOIN levels l ON up.level_id = l.id
                    JOIN difficulty d ON l.difficulty_id = d.id
                    WHERE up.user_id = %s
                    ORDER BY d.type, l.number
                    """,
                    (user_id_db,),
                )
                results = cursor.fetchall()
                progress_list = []
                for row in results:
                    (
                        progress_id,
                        uid,
                        lid,
                        status,
                        reason,
                        matrix_json,
                        updated_at,
                        level_num,
                        diff_type,
                    ) = row
                    matrix_data = matrix_json
                    if isinstance(matrix_json, str):
                        matrix_data = json.loads(matrix_json)
                    progress_list.append(
                        {
                            "id": progress_id,
                            "username": username,
                            "level": level_num,
                            "difficulty": diff_type,
                            "status": status,
                            "reason": reason,
                            "matrix": matrix_data,
                            "updated_at": updated_at.isoformat() if updated_at else None,
                        }
                    )
                return {"success": True, "progress": progress_list}

        # Legacy format: user_id-based
        elif user_id is not None:
            if level_id is not None:
                cursor.execute(
                    """
                    SELECT id, user_id, level_id, status, reason, matrix, updated_at
                    FROM user_progress
                    WHERE user_id = %s AND level_id = %s
                    """,
                    (user_id, level_id),
                )
                result = cursor.fetchone()
                if result is None:
                    raise HTTPException(
                        status_code=404, detail=f"Progress not found for user_id {user_id}, level_id {level_id}"
                    )
                progress_id, uid, lid, status, reason, matrix_json, updated_at = result
                matrix_data = matrix_json
                if isinstance(matrix_json, str):
                    matrix_data = json.loads(matrix_json)
                return {
                    "success": True,
                    "progress": {
                        "id": progress_id,
                        "user_id": uid,
                        "level_id": lid,
                        "status": status,
                        "reason": reason,
                        "matrix": matrix_data,
                        "updated_at": updated_at.isoformat() if updated_at else None,
                    },
                }
            else:
                cursor.execute(
                    """
                    SELECT id, user_id, level_id, status, reason, matrix, updated_at
                    FROM user_progress
                    WHERE user_id = %s
                    ORDER BY level_id
                    """,
                    (user_id,),
                )
                results = cursor.fetchall()
                progress_list = []
                for row in results:
                    progress_id, uid, lid, status, reason, matrix_json, updated_at = row
                    matrix_data = matrix_json
                    if isinstance(matrix_json, str):
                        matrix_data = json.loads(matrix_json)
                    progress_list.append(
                        {
                            "id": progress_id,
                            "user_id": uid,
                            "level_id": lid,
                            "status": status,
                            "reason": reason,
                            "matrix": matrix_data,
                            "updated_at": updated_at.isoformat() if updated_at else None,
                        }
                    )
                return {"success": True, "progress": progress_list}
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'username' or 'user_id' must be provided",
            )
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

