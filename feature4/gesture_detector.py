
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
    """
    Feature 4 gestures:
      Open palm       -> PAUSE_AUTO
      Both hands up   -> CLIMAX
      Point direction -> SPOTLIGHT_DIRECTION
      Arms spread     -> FULL_STAGE_REVEAL
    """

    def __init__(self):
        for path in (HAND_MODEL, POSE_MODEL):
            if not path.is_file():
                raise FileNotFoundError(f"Model file not found: {path}")

        hand_options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(HAND_MODEL)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.hands = vision.GestureRecognizer.create_from_options(
            hand_options
        )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(POSE_MODEL)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            pose_options
        )

        self.last_timestamp = 0
        self.active_pose = None
        self.pose_frames = 0
        self.last_event = None
        self.direction_last_sent = 0.0

    @staticmethod
    def _visible(landmark, threshold=0.50):
        return getattr(landmark, "visibility", 1.0) >= threshold

    def _get_pose(self, rgb, timestamp):
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        return self.pose.detect_for_video(image, timestamp)

    def _get_hands(self, rgb, timestamp):
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        return self.hands.recognize_for_video(image, timestamp)

    def _classify_pose(self, landmarks):
        # MediaPipe Pose landmark indices.
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

        if not all(self._visible(p) for p in required):
            return None, None

        shoulder_width = max(
            abs(right_shoulder.x - left_shoulder.x),
            0.08,
        )

        # --------------------------------------------------
        # 1. CLIMAX: both wrists clearly above the head.
        # This is checked before pointing.
        # --------------------------------------------------
        head_y = min(
            nose.y,
            left_shoulder.y,
            right_shoulder.y,
        )

        both_hands_up = (
            left_wrist.y < head_y - 0.08
            and right_wrist.y < head_y - 0.08
        )

        if both_hands_up:
            return "CLIMAX", None

        # --------------------------------------------------
        # 2. FULL_STAGE_REVEAL: both arms extended sideways.
        # Absolute horizontal distances work with mirrored
        # and non-mirrored previews.
        # --------------------------------------------------
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

        arms_spread = (
            left_dx > shoulder_width * 0.90
            and right_dx > shoulder_width * 0.90
            and left_level
            and right_level
        )

        if arms_spread:
            return "FULL_STAGE_REVEAL", None

        # --------------------------------------------------
        # 3. SPOTLIGHT_DIRECTION: one arm extended.
        # Do not treat a two-handed overhead pose as pointing.
        # --------------------------------------------------
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

            # A raised hand close to vertical is not enough
            # by itself to count as pointing.
            clearly_horizontal = (
                abs(dx) > shoulder_width * 0.80
                and abs(dx) > abs(dy) * 0.80
            )
            clearly_upward = (
                dy > shoulder_width * 0.65
                and dy > abs(dx) * 0.60
            )

            if clearly_horizontal or clearly_upward:
                candidates.append((dx, dy, length))

        if candidates:
            dx, dy, length = max(
                candidates,
                key=lambda item: item[2],
            )

            return "SPOTLIGHT_DIRECTION", {
                "x": round(dx / length, 3),
                "y": round(dy / length, 3),
            }

        return None, None

    def detect(self, frame):
        """
        Returns an event dictionary or None.

        Static gestures require four stable frames.
        Spotlight direction is emitted repeatedly while pointing.
        """
        now = time.monotonic()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = max(
            int(now * 1000),
            self.last_timestamp + 1,
        )
        self.last_timestamp = timestamp

        pose_result = self._get_pose(rgb, timestamp)
        hand_result = self._get_hands(rgb, timestamp)

        pose_name = None
        direction = None

        if pose_result.pose_landmarks:
            pose_name, direction = self._classify_pose(
                pose_result.pose_landmarks[0]
            )

        # Recognize open palm only when the hand classifier
        # has sufficient confidence.
        open_palm = False

        if hand_result.gestures:
            for categories in hand_result.gestures:
                if not categories:
                    continue

                category = categories[0]

                if (
                    category.category_name == "Open_Palm"
                    and category.score >= 0.70
                ):
                    open_palm = True
                    break

        # Explicit priority:
        # 1. Two-arm stage poses
        # 2. Directional pointing
        # 3. Open-palm pause
        if pose_name in ("CLIMAX", "FULL_STAGE_REVEAL"):
            current = pose_name
        elif pose_name == "SPOTLIGHT_DIRECTION":
            current = pose_name
        elif open_palm:
            current = "PAUSE_AUTO"
        else:
            current = None

        # Spotlight direction is continuous, but only while
        # the pose classifier reports pointing.
        if current == "SPOTLIGHT_DIRECTION" and direction:
            self.active_pose = current
            self.pose_frames = 0
            self.last_event = current

            if now - self.direction_last_sent >= 0.10:
                self.direction_last_sent = now
                return {
                    "type": "GESTURE_EVENT",
                    "event": "SPOTLIGHT_DIRECTION",
                    "direction": direction,
                }

            return None

        # Reset after releasing a gesture.
        if current is None:
            self.active_pose = None
            self.pose_frames = 0
            self.last_event = None
            return None

        # Require four consecutive frames for static gestures.
        if current == self.active_pose:
            self.pose_frames += 1
        else:
            self.active_pose = current
            self.pose_frames = 1

        if self.pose_frames < 4:
            return None

        # Emit each static gesture only once while held.
        if current == self.last_event:
            return None

        self.last_event = current

        return {
            "type": "GESTURE_EVENT",
            "event": current,
        }

    def close(self):
        self.hands.close()
        self.pose.close()
