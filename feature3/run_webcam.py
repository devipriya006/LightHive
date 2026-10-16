
from __future__ import annotations

import json
import os
import time

import cv2
import torch

from feature3.audience_energy import AudienceEnergyAnalyzer


def main() -> None:
    # Configure CPU threads before running inference.
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch may already have initialized this setting.
        pass

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Fallback if DirectShow cannot open the webcam.
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions "
            "or try another camera index."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    analyzer = None

    try:
        analyzer = AudienceEnergyAnalyzer(
            model_path="yolo11n-pose.pt",
            image_size=320,
            max_people=10,
            smoothing=0.25,
            confidence=0.30,
            device="cpu",
        )

        last_log = 0.0
        last_frame_time = time.perf_counter()
        fps = 0.0

        window_name = "NeuralStage AI - Audience Tracking"

        # Create the video window only once, outside the loop.
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        print("NeuralStage AI - Feature 3")
        print("Multi-person tracking and audience energy detection")
        print("Press Q in the video window to exit.")

        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            now = time.perf_counter()
            elapsed = max(now - last_frame_time, 1e-6)
            instant_fps = 1.0 / elapsed

            fps = (
                instant_fps
                if fps == 0.0
                else 0.9 * fps + 0.1 * instant_fps
            )
            last_frame_time = now

            annotated, payload = analyzer.update(frame)

            # Display audience energy.
            cv2.putText(
                annotated,
                f"Audience Energy: "
                f"{payload['audience_energy']:.1f}/100",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Display detected people and activity level.
            cv2.putText(
                annotated,
                f"People: {payload['person_count']} | "
                f"Level: {payload['activity_level']}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            # Display approximate processing FPS.
            cv2.putText(
                annotated,
                f"CPU FPS: {fps:.1f}",
                (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            # Print JSON for later integration with Node.js/Unity.
            if now - last_log >= 0.5:
                print(json.dumps(payload), flush=True)
                last_log = now

            cv2.imshow(window_name, annotated)

            # Press Q while the video window is focused to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if analyzer is not None:
            analyzer.close()

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
