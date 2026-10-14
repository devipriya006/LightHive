
import json
import cv2

from feature4.gesture_detector import GestureDetector


def main():
    detector = GestureDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        detector.close()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    print("Feature 4 running. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read webcam frame.")
                break

            event = detector.detect(frame)

            if event:
                print(json.dumps(event))

                label = event["event"]
                if label == "SPOTLIGHT_DIRECTION":
                    d = event["direction"]
                    label += f"  x={d['x']} y={d['y']}"

                cv2.putText(
                    frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("NeuralStage AI - Feature 4", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()

