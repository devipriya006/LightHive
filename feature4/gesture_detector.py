
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parent.parent

HAND_MODEL = ROOT / "models" / "feature4" / "gesture_recognizer.task"
POSE_MODEL = ROOT / "models" / "feature4" / "pose_landmarker_lite.task"


class GestureDetector:
    """Detect Auto Pause, Climax, Spotlight Direction and Full Stage Reveal."""

    def __init__(self):
        for path in (HAND_MODEL, POSE_MODEL):
            if not path.is_file():
                raise FileNotFoundError(f"Model file not found: {path}")

        self.hands = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(HAND_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )

        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(POSE_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )

        self.last_timestamp = 0
        self.active_pose = None
        self.pose_frames = 0
        self.last_event = None
        self.direction_last_sent = 0.0
        self.pause_active = False

    @staticmethod
    def _visible(landmark, threshold=0.50):
        return getattr(landmark, "visibility", 1.0) >= threshold

    def _get_pose(self, rgb, timestamp):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.pose.detect_for_video(image, timestamp)

    def _get_hands(self, rgb, timestamp):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.hands.recognize_for_video(image, timestamp)

    def _classify_pose(self, landmarks):
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        required = [
            left_shoulder, right_shoulder,
            left_elbow, right_elbow,
            left_wrist, right_wrist,
        ]

        if not all(self._visible(point) for point in required):
            return None, None

        shoulder_width = max(
            abs(right_shoulder.x - left_shoulder.x), 0.08
        )

        # CLIMAX: both wrists are clearly above the head.
        head_y = min(nose.y, left_shoulder.y, right_shoulder.y)

        if (
            left_wrist.y < head_y - 0.08
            and right_wrist.y < head_y - 0.08
        ):
            return "CLIMAX", None

        # FULL_STAGE_REVEAL: both arms extend sideways.
        left_dx = abs(left_wrist.x - left_shoulder.x)
        right_dx = abs(right_wrist.x - right_shoulder.x)

        left_level = (
            abs(left_wrist.y - left_shoulder.y)
            < shoulder_width * 0.90
        )
        right_level = (
            abs(right_wrist.y - right_shoulder.y)
            < shoulder_width * 0.90
        )

        if (
            left_dx > shoulder_width * 0.90
            and right_dx > shoulder_width * 0.90
            and left_level
            and right_level
        ):
            return "FULL_STAGE_REVEAL", None

        # SPOTLIGHT_DIRECTION: one arm extends sideways or upward.
        candidates = []

        for shoulder, elbow, wrist in (
            (left_shoulder, left_elbow, left_wrist),
            (right_shoulder, right_elbow, right_wrist),
        ):
            dx = wrist.x - shoulder.x
            dy = shoulder.y - wrist.y
            length = (dx * dx + dy * dy) ** 0.5

            if length < shoulder_width * 1.10:
                continue

            horizontal = (
                abs(dx) > shoulder_width * 0.80
                and abs(dx) > abs(dy) * 0.80
            )
            upward = (
                dy > shoulder_width * 0.65
                and dy > abs(dx) * 0.60
            )

            if horizontal or upward:
                candidates.append((dx, dy, length))

        if candidates:
            dx, dy, length = max(candidates, key=lambda item: item[2])
            return "SPOTLIGHT_DIRECTION", {
                "x": round(dx / length, 3),
                "y": round(dy / length, 3),
            }

        return None, None

    def detect(self, frame):
        now = time.monotonic()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = max(int(now * 1000), self.last_timestamp + 1)
        self.last_timestamp = timestamp

        pose_result = self._get_pose(rgb, timestamp)
        hand_result = self._get_hands(rgb, timestamp)

        pose_name = None
        direction = None

        if pose_result.pose_landmarks:
            pose_name, direction = self._classify_pose(
                pose_result.pose_landmarks[0]
            )

        # One detected open palm is enough to pause.
        one_hand_open = False

        if hand_result.gestures:
            for categories in hand_result.gestures:
                if any(
                    category.category_name == "Open_Palm"
                    and category.score >= 0.70
                    for category in categories
                ):
                    one_hand_open = True
                    break

        # Highest priority: PAUSE_AUTO overrides every pose gesture.
        self.pause_active = one_hand_open

        if self.pause_active:
            self.active_pose = None
            self.pose_frames = 0
            self.last_event = None

            return {
                "type": "GESTURE_STATE",
                "event": "PAUSE_AUTO",
                "pause_active": True,
                "direction": None,
            }

        # The palm has closed or is no longer detected. Resume processing.
        if pose_name == "SPOTLIGHT_DIRECTION" and direction:
            self.active_pose = pose_name
            self.pose_frames = 0
            self.last_event = pose_name

            if now - self.direction_last_sent >= 0.10:
                self.direction_last_sent = now
                return {
                    "type": "GESTURE_EVENT",
                    "event": "SPOTLIGHT_DIRECTION",
                    "direction": direction,
                    "pause_active": False,
                }

            return {
                "type": "GESTURE_STATE",
                "event": None,
                "direction": direction,
                "pause_active": False,
            }

        if pose_name is None:
            self.active_pose = None
            self.pose_frames = 0
            self.last_event = None

            return {
                "type": "GESTURE_STATE",
                "event": None,
                "direction": None,
                "pause_active": False,
            }

        if pose_name == self.active_pose:
            self.pose_frames += 1
        else:
            self.active_pose = pose_name
            self.pose_frames = 1

        if self.pose_frames < 4:
            return {
                "type": "GESTURE_STATE",
                "event": None,
                "direction": None,
                "pause_active": False,
            }

        if pose_name == self.last_event:
            return {
                "type": "GESTURE_STATE",
                "event": None,
                "direction": None,
                "pause_active": False,
            }

        self.last_event = pose_name

        return {
            "type": "GESTURE_EVENT",
            "event": pose_name,
            "direction": None,
            "pause_active": False,
        }

    def close(self):
        self.hands.close()
        self.pose.close()
