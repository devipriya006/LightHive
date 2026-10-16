
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackHistory:
    """Recent movement state for one tracked person."""

    center: np.ndarray
    wrists: np.ndarray | None
    torso_y: float | None
    updated_at: float
    activity: float = 0.0
    hand_activity: float = 0.0
    body_activity: float = 0.0
    vertical_activity: float = 0.0


class AudienceEnergyAnalyzer:
    """
    Multi-person audience activity estimator.

    Uses a YOLO pose model and persistent ByteTrack tracking IDs.
    Activity scores are motion estimates, not definitive classifications
    of emotions or specific actions such as jumping.
    """

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        image_size: int = 416,
        max_people: int = 10,
        smoothing: float = 0.75,
        confidence: float = 0.30,
        device: str = "cpu",
    ) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1].")
        if image_size < 320:
            raise ValueError("image_size must be at least 320.")
        if max_people < 1:
            raise ValueError("max_people must be positive.")

        self.model = YOLO(model_path)
        self.image_size = image_size
        self.max_people = max_people
        self.smoothing = smoothing
        self.confidence = confidence
        self.device = device

        self.histories: dict[int, TrackHistory] = {}
        self._energy = 0.0
        self._last_frame_at: float | None = None

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _level(score: float) -> str:
        if score < 20:
            return "low"
        if score < 60:
            return "moderate"
        return "high"

    def _smooth(self, previous: float, current: float) -> float:
        a = self.smoothing
        return (1.0 - a) * previous + a * current

    def update(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Analyze one webcam frame and return an annotated frame and payload."""
        if frame is None or frame.size == 0:
            raise ValueError("Received an empty video frame.")

        now = monotonic()
        frame_start = perf_counter()
        inference_start = perf_counter()

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],              # COCO class 0 = person
            conf=self.confidence,
            iou=0.5,
            imgsz=self.image_size,
            max_det=self.max_people,
            device=self.device,
            verbose=False,
        )
        inference_ms = (perf_counter() - inference_start) * 1000.0
        result = results[0]
        annotated = result.plot()
        people: list[dict[str, Any]] = []
        visible_ids: set[int] = set()

        boxes = result.boxes
        keypoints = result.keypoints

        if (
            boxes is not None
            and len(boxes) > 0
            and boxes.id is not None
            and keypoints is not None
            and keypoints.xy is not None
        ):
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            poses = keypoints.xy.cpu().numpy()

            keypoint_conf = (
                keypoints.conf.cpu().numpy()
                if keypoints.conf is not None
                else None
            )

            for index, raw_id in enumerate(ids):
                track_id = int(raw_id)
                visible_ids.add(track_id)

                x1, y1, x2, y2 = map(float, xyxy[index])
                box_width = max(x2 - x1, 1.0)
                box_height = max(y2 - y1, 1.0)
                body_scale = max(float(np.hypot(box_width, box_height)), 1.0)

                pose = poses[index]

                # COCO keypoints: shoulders 5/6, wrists 9/10, hips 11/12.
                # Ignore low-confidence wrist landmarks.
                wrists: np.ndarray | None = None
                if len(pose) >= 13:
                    wrist_indices = [9, 10]
                    if keypoint_conf is None:
                        valid_wrists = np.ones(2, dtype=bool)
                    else:
                        valid_wrists = keypoint_conf[index, wrist_indices] >= 0.35

                    if np.any(valid_wrists):
                        wrists = pose[wrist_indices][valid_wrists].copy()   

                torso_y: float | None = None
                if len(pose) >= 13:
                    torso_indices = [5, 6, 11, 12]
                    if keypoint_conf is None:
                        valid_torso = np.ones(4, dtype=bool)
                    else:
                        valid_torso = keypoint_conf[index, torso_indices] >= 0.35

                    if np.any(valid_torso):
                        torso_y = float(
                            pose[torso_indices][valid_torso, 1].mean()
                        )

                center = np.array(
                    [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    dtype=np.float32,
                )

                previous = self.histories.get(track_id)
                body_raw = 0.0
                hand_raw = 0.0
                vertical_raw = 0.0

                if previous is not None:
                    dt = max(now - previous.updated_at, 1.0 / 60.0)

                    # Normalize movement by person size so distant people
                    # are not automatically assigned lower activity.
                    center_speed = (
                        float(np.linalg.norm(center - previous.center))
                        / body_scale
                        / dt
                    )
                    body_raw = self._clamp01(center_speed / 0.30)

                    
                    if (
                        wrists is not None
                        and previous.wrists is not None
                        and wrists.shape == previous.wrists.shape
                    ):
                        wrist_displacements = np.linalg.norm(
                            wrists - previous.wrists, axis=1
                        )

                        wrist_speed = (
                            float(np.mean(wrist_displacements))
                            / body_scale
                            / dt
                        )

                        hand_raw = self._clamp01(wrist_speed / 0.25)


                    if torso_y is not None and previous.torso_y is not None:
                        vertical_speed = (
                            abs(torso_y - previous.torso_y)
                            / body_scale
                            / dt
                        )
                        vertical_raw = self._clamp01(vertical_speed / 0.25)

                    body_activity = self._smooth(
                        previous.body_activity, body_raw
                    )
                    hand_activity = self._smooth(
                        previous.hand_activity, hand_raw
                    )
                    vertical_activity = self._smooth(
                        previous.vertical_activity, vertical_raw
                    )
                    activity = (
                     0.30 * body_activity
                    + 0.45 * hand_activity
                     + 0.25 * vertical_activity
                    )
                    activity = self._smooth(previous.activity, activity)
                else:
                    body_activity = 0.0
                    hand_activity = 0.0
                    vertical_activity = 0.0
                    activity = 0.0

                self.histories[track_id] = TrackHistory(
                    center=center,
                    wrists=wrists,
                    torso_y=torso_y,
                    updated_at=now,
                    activity=activity,
                    hand_activity=hand_activity,
                    body_activity=body_activity,
                    vertical_activity=vertical_activity,
                )

                people.append({
                    "track_id": track_id,
                    "bbox_xyxy": [
                        round(x1, 1), round(y1, 1),
                        round(x2, 1), round(y2, 1),
                    ],
                    "activity_score": round(activity * 100.0, 1),
                    "hand_activity": round(hand_activity, 3),
                    "body_motion": round(body_activity, 3),
                    "vertical_motion": round(vertical_activity, 3),
                })

        # Keep recently lost IDs briefly so a short occlusion does not
        # immediately erase movement history. Remove stale IDs after 2 s.
        stale_ids = [
            track_id
            for track_id, history in self.histories.items()
            if now - history.updated_at > 2.0
        ]
        for track_id in stale_ids:
            del self.histories[track_id]

        # Use visible people only. Avoid counting stale, hidden tracks.
        raw_energy = (
            float(np.mean([p["activity_score"] for p in people]))
            if people else 0.0
        )
        self._energy = self._smooth(self._energy, raw_energy)

        payload: dict[str, Any] = {
            "event": "audience_energy",
            "timestamp_monotonic": round(now, 3),
            "audience_energy": round(self._energy, 2),
            "activity_level": self._level(self._energy),
            "person_count": len(people),
            "people": people,
            "inference_ms": round(inference_ms, 1),
            "processing_ms": round((perf_counter() - frame_start) * 1000.0, 1),
        }

        return annotated, payload

    def close(self) -> None:
        """Release resources if supported by the loaded model."""
        self.histories.clear()
