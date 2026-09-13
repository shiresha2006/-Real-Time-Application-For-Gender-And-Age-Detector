"""
Local, no-Streamlit entry point. Use this to record your demo video:
it opens a plain OpenCV window, works with a live webcam or a video
file, and can optionally save the annotated output to disk.

Usage:
    python run_webcam.py                      # default webcam (index 0)
    python run_webcam.py --source path/to.mp4 # run on a video file
    python run_webcam.py --save out.mp4        # also write annotated video
    python run_webcam.py --source 1             # a different camera index

Press 'q' to quit, 's' to export the session CSV log at any time.
"""

from __future__ import annotations
import argparse
import cv2

from src.pipeline import RealtimePipeline


def main():
    parser = argparse.ArgumentParser(description="Real-time age & gender detector (local)")
    parser.add_argument("--source", default="0", help="Webcam index or video file path (default: 0)")
    parser.add_argument("--save", default=None, help="Optional path to save annotated output video")
    parser.add_argument("--csv-out", default="session_log.csv", help="Path to write the session CSV on quit/save")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    pipeline = RealtimePipeline()
    writer = None

    print("Running. Press 'q' to quit, 's' to save the session CSV log.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, _ = pipeline.process_frame(frame)

        if args.save and writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w, h))
        if writer is not None:
            writer.write(annotated)

        cv2.imshow("Real-Time Age & Gender Detector", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            with open(args.csv_out, "wb") as f:
                f.write(pipeline.session_log.to_csv_bytes())
            print(f"Session log saved to {args.csv_out}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    with open(args.csv_out, "wb") as f:
        f.write(pipeline.session_log.to_csv_bytes())
    print(f"Final session log saved to {args.csv_out}")
    print("Summary:", pipeline.session_log.summary())


if __name__ == "__main__":
    main()
