"""Settings loaded from a local .env file (see .env.example).

A tiny built-in parser is used so the project does not need python-dotenv.
Real environment variables always win over values in .env.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


_load_env_file(BASE_DIR / ".env")


def _get(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"[config] Invalid value for {name}={raw!r}, using default {default!r}")
        return default


def _bool(raw):
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(raw)


def _path(raw):
    p = Path(raw)
    return p if p.is_absolute() else BASE_DIR / p


# ---------------- Window / game ----------------
WINDOW_WIDTH = _get("WINDOW_WIDTH", 1000, int)
WINDOW_HEIGHT = _get("WINDOW_HEIGHT", 600, int)
FPS = _get("FPS", 60, int)
SOUND_ENABLED = _get("SOUND_ENABLED", True, _bool)
SOUND_VOLUME = _get("SOUND_VOLUME", 0.5, float)
SAVE_FILE = _get("SAVE_FILE", BASE_DIR / "savegame.json", _path)

# ---------------- Hand gestures ----------------
CAMERA_INDEX = _get("CAMERA_INDEX", 0, int)
SHOW_CAMERA_PREVIEW = _get("SHOW_CAMERA_PREVIEW", True, _bool)
GESTURE_MIN_CONFIDENCE = _get("GESTURE_MIN_CONFIDENCE", 0.6, float)
GESTURE_SMOOTHING_FRAMES = _get("GESTURE_SMOOTHING_FRAMES", 5, int)
HAND_MODEL_PATH = _get("HAND_MODEL_PATH", BASE_DIR / "models" / "hand_landmarker.task", _path)

# ---------------- Reinforcement learning ----------------
MODEL_PATH = _get("MODEL_PATH", BASE_DIR / "models" / "q_table.json", _path)
TRAIN_EPISODES = _get("TRAIN_EPISODES", 2000, int)
LEARNING_RATE = _get("LEARNING_RATE", 0.1, float)
DISCOUNT = _get("DISCOUNT", 0.98, float)
EPSILON_START = _get("EPSILON_START", 1.0, float)
EPSILON_END = _get("EPSILON_END", 0.02, float)
EPSILON_DECAY_EPISODES = _get("EPSILON_DECAY_EPISODES", 1500, int)
TRAIN_SEED = _get("TRAIN_SEED", 42, int)
MAX_EPISODE_SECONDS = _get("MAX_EPISODE_SECONDS", 90, float)
TRAINING_LOG = _get("TRAINING_LOG", BASE_DIR / "models" / "training_log.csv", _path)
