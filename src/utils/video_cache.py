import json
import os
import hashlib
from typing import Dict, List, Tuple


CACHE_DIR = os.path.join("data", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "processed_videos.json")
METADATA_DIR = os.path.join("data", "metadata")


def _ensure_directories() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)


def _empty_cache() -> Dict[str, Dict]:
    return {"videos": {}}


def video_id_for_path(path: str) -> str:
    """Create a stable identifier for a video path."""
    absolute_path = os.path.abspath(path)
    digest = hashlib.sha256(absolute_path.encode("utf-8")).hexdigest()
    # Truncate for readability while keeping collision risk low.
    return digest[:16]


def load_cache() -> Dict[str, Dict]:
    _ensure_directories()
    if not os.path.exists(CACHE_FILE):
        return _empty_cache()

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict) or "videos" not in data:
                return _empty_cache()
            return data
    except (json.JSONDecodeError, OSError):
        return _empty_cache()


def save_cache(cache: Dict[str, Dict]) -> None:
    _ensure_directories()
    with open(CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)


def register_videos(video_paths: List[str]) -> Dict[str, Dict]:
    """Ensure cache entries exist for the provided video paths and return the cache."""
    cache = load_cache()
    videos = cache.setdefault("videos", {})

    for path in video_paths:
        if not path:
            continue
        absolute_path = os.path.abspath(path)
        video_id = video_id_for_path(absolute_path)
        entry = videos.setdefault(video_id, {})
        entry.update(
            {
                "path": absolute_path,
                "filename": os.path.basename(absolute_path),
                "metadata_file": entry.get("metadata_file") or _default_metadata_path(video_id),
            }
        )

    save_cache(cache)
    return cache


def _default_metadata_path(video_id: str) -> str:
    return os.path.join(METADATA_DIR, f"{video_id}_objects.json")


def get_videos_to_process(video_paths: List[str]) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Return videos that need processing and the refreshed cache."""
    cache = register_videos(video_paths)
    videos = cache.get("videos", {})
    to_process: List[Dict] = []

    for path in video_paths:
        if not path:
            continue
        absolute_path = os.path.abspath(path)
        video_id = video_id_for_path(absolute_path)
        entry = videos.get(video_id, {})
        metadata_file = entry.get("metadata_file") or _default_metadata_path(video_id)
        current_mtime = _safe_mtime(absolute_path)
        cached_mtime = entry.get("mtime")

        if (
            current_mtime is None
            or not os.path.exists(metadata_file)
            or cached_mtime is None
            or abs(cached_mtime - current_mtime) > 1e-6
        ):
            to_process.append(
                {
                    "id": video_id,
                    "path": absolute_path,
                    "filename": os.path.basename(absolute_path),
                    "metadata_file": metadata_file,
                    "mtime": current_mtime,
                }
            )

    return to_process, cache


def _safe_mtime(path: str):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def update_video_entry(video_id: str, *, metadata_file: str = None, mtime: float = None) -> None:
    cache = load_cache()
    videos = cache.setdefault("videos", {})
    entry = videos.setdefault(video_id, {})

    if metadata_file:
        entry["metadata_file"] = metadata_file
    if mtime is not None:
        entry["mtime"] = mtime

    save_cache(cache)


def iter_registered_videos():
    cache = load_cache()
    for video_id, entry in cache.get("videos", {}).items():
        yield video_id, entry


def get_video_entry(video_id: str) -> Dict:
    cache = load_cache()
    return cache.get("videos", {}).get(video_id, {})




