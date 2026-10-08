"""Webcam hand-gesture controls (OpenCV + MediaPipe Hand Landmarker).

    open palm (3+ fingers up)  -> GAS
    fist (0-1 fingers up)      -> BRAKE
    two fingers / no hand      -> COAST

The camera runs on a background thread so the game keeps its frame rate.
The MediaPipe hand model (~8 MB) is downloaded on first use.
"""
import threading
import time
import urllib.request
from collections import Counter, deque

from hill_env import BRAKE, COAST, GAS

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
PREVIEW_SIZE = (224, 168)

WRIST = 0
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, middle joint) for index..pinky
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
LABELS = {GAS: "OPEN PALM = GAS", BRAKE: "FIST = BRAKE", COAST: "COAST"}


class GestureError(RuntimeError):
    """Raised with a user-friendly message when gestures cannot start."""


def count_extended_fingers(landmarks):
    """A finger is up when its tip is clearly farther from the wrist than its middle joint."""
    wx, wy = landmarks[WRIST].x, landmarks[WRIST].y
    up = 0
    for tip, pip in FINGERS:
        tip_d = (landmarks[tip].x - wx) ** 2 + (landmarks[tip].y - wy) ** 2
        pip_d = (landmarks[pip].x - wx) ** 2 + (landmarks[pip].y - wy) ** 2
        if tip_d > pip_d * 1.3:
            up += 1
    return up


def classify(landmarks):
    fingers = count_extended_fingers(landmarks)
    if fingers >= 3:
        return GAS
    if fingers <= 1:
        return BRAKE
    return COAST


class GestureController:
    def __init__(self, camera_index=0, min_confidence=0.6, smoothing_frames=5,
                 model_path="models/hand_landmarker.task", preview=True):
        self.camera_index = camera_index
        self.min_confidence = min_confidence
        self.model_path = model_path
        self.preview_enabled = preview
        self._history = deque(maxlen=max(1, smoothing_frames))
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._cap = None
        self._landmarker = None
        self.action = COAST
        self.hand_visible = False
        self.fps = 0.0
        self._preview = None

    # ---------------- lifecycle ----------------
    def start(self):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise GestureError(
                "Hand gestures need OpenCV and MediaPipe:  pip install opencv-python mediapipe"
            ) from exc
        self._cv2, self._mp = cv2, mp

        self._ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.camera_index, backend)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self._landmarker.close()
            raise GestureError(f"Could not open webcam #{self.camera_index}. Check CAMERA_INDEX in .env")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="gestures", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self.action = COAST
        self.hand_visible = False

    def _ensure_model(self):
        path = self.model_path
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {path} ...")
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
            tmp.replace(path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise GestureError(f"Could not download the hand model (internet needed once): {exc}") from exc

    # ---------------- camera thread ----------------
    def _loop(self):
        cv2, mp = self._cv2, self._mp
        t0 = time.monotonic()
        last_ts = -1
        frames, fps_t = 0, time.monotonic()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)  # mirror, so moving your hand right moves it right on screen
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(last_ts + 1, int((time.monotonic() - t0) * 1000))
            last_ts = ts
            result = self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
            self._history.append(classify(landmarks) if landmarks else COAST)
            self.action = Counter(self._history).most_common(1)[0][0]
            self.hand_visible = landmarks is not None

            if self.preview_enabled:
                small = cv2.resize(rgb, PREVIEW_SIZE)
                if landmarks:
                    self._draw_hand(small, landmarks)
                with self._lock:
                    self._preview = small.tobytes()

            frames += 1
            now = time.monotonic()
            if now - fps_t >= 1.0:
                self.fps = frames / (now - fps_t)
                frames, fps_t = 0, now

    def _draw_hand(self, img, landmarks):
        cv2 = self._cv2
        w, h = PREVIEW_SIZE
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        color = {GAS: (60, 220, 90), BRAKE: (240, 70, 60), COAST: (240, 200, 60)}[self.action]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(img, p, 3, (255, 255, 255), -1)

    def get_preview(self):
        """Latest camera frame as raw RGB bytes (PREVIEW_SIZE), or None."""
        with self._lock:
            return self._preview

    @property
    def label(self):
        return LABELS[self.action] if self.hand_visible else "NO HAND - COAST"
