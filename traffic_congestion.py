"""
Traffic Congestion Detection with YOLO11 + ByteTrack

"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


# COCO vehicle classes (Ultralytics COCO class IDs)
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
VEHICLE_CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def default_roi(w: int, h: int) -> np.ndarray:
    """
    Trapezoid ROI covering the road/lane area.
    Adjust these points for your camera view.
    """
    return np.array([
        [int(0.50 * w), int(0.80 * h)],  # bottom left
        [int(0.52 * w), int(0.45 * h)],  # top left
        [int(0.77 * w), int(0.45 * h)],  # top right
        [int(0.99 * w), int(0.80 * h)],  # bottom right
    ], dtype=np.int32)



def point_in_poly(pt, poly) -> bool:
    # cv2.pointPolygonTest returns >0 inside, 0 on edge, <0 outside
    return cv2.pointPolygonTest(poly, pt, False) >= 0


class TrafficCongestionDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        tracker_path: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.7,
        speed_history: int = 5,
        min_tracks_for_congestion: int = 8,
        max_avg_speed_px: float = 12.0,
        congestion_score_threshold: float = 0.55,
    ):
        self.model = YOLO(model_path)
        self.tracker_path = tracker_path
        self.conf = conf
        self.iou = iou

        self.speed_history = speed_history
        self.min_tracks_for_congestion = min_tracks_for_congestion
        self.max_avg_speed_px = max_avg_speed_px
        self.congestion_score_threshold = congestion_score_threshold

        self.track_history = defaultdict(lambda: deque(maxlen=speed_history))
        self.last_seen = {}

    def _estimate_speed(self, track_id: int) -> float:
        hist = self.track_history[track_id]
        if len(hist) < 2:
            return 0.0
        p1 = np.array(hist[-2], dtype=np.float32)
        p2 = np.array(hist[-1], dtype=np.float32)
        return float(np.linalg.norm(p2 - p1))

    def process_frame(self, frame: np.ndarray, roi: np.ndarray):
        """
        Returns:
            annotated_frame, stats_dict
        """
        # ByteTrack via Ultralytics tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_path,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        annotated = results.plot()

        active_vehicle_tracks = 0
        speeds = []
        per_class_counts = defaultdict(int)

        # Draw ROI
        cv2.polylines(annotated, [roi], isClosed=True, color=(0, 255, 255), thickness=2)

        if results.boxes is not None and results.boxes.id is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            class_ids = results.boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes_xyxy, track_ids, class_ids):
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                x1, y1, x2, y2 = box
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                centroid = (cx, cy)

                self.track_history[track_id].append(centroid)
                self.last_seen[track_id] = time.time()

                if point_in_poly(centroid, roi):
                    active_vehicle_tracks += 1
                    per_class_counts[cls_id] += 1

                    speed_px = self._estimate_speed(track_id)
                    speeds.append(speed_px)

                    # Draw track info
                    label = f"ID {track_id} {VEHICLE_CLASS_NAMES.get(cls_id, str(cls_id))}"
                    cv2.putText(
                        annotated,
                        label,
                        (int(x1), max(0, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # Simple congestion score:
        # - more vehicles => more congestion
        # - lower speed => more congestion
        density_score = min(active_vehicle_tracks / max(self.min_tracks_for_congestion, 1), 1.0)
        speed_score = 1.0 - min(avg_speed / max(self.max_avg_speed_px, 1e-6), 1.0)
        congestion_score = 0.6 * density_score + 0.4 * speed_score

        congested = (
            active_vehicle_tracks >= self.min_tracks_for_congestion
            and congestion_score >= self.congestion_score_threshold
        )

        state_text = "CONGESTED" if congested else "FLOWING"
        state_color = (0, 0, 255) if congested else (0, 255, 0)

        # Overlay
        cv2.rectangle(annotated, (10, 10), (430, 150), (0, 0, 0), -1)
        cv2.putText(annotated, f"State: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
        cv2.putText(
            annotated,
            f"Vehicles in ROI: {active_vehicle_tracks}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        cv2.putText(
            annotated,
            f"Congestion score: {congestion_score:.2f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        stats = {
            "congested": congested,
            "state_text": state_text,
            "active_vehicle_tracks": active_vehicle_tracks,
            "congestion_score": congestion_score,
            "per_class_counts": dict(per_class_counts),
        }
        return annotated, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video path or webcam index (e.g. 0)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO11 model path")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--save", default=True ,action="store_true", help="Save output video")
    parser.add_argument("--output", type=str, default="/content/drive/MyDrive/output.mp4", help="Output video path")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    detector = TrafficCongestionDetector(
        model_path=args.model,
        tracker_path=args.tracker,
        conf=args.conf,
        iou=args.iou,
    )

    writer = None
    roi = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if roi is None:
            h, w = frame.shape[:2]
            roi = default_roi(w, h)

            if args.save:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

        annotated, stats = detector.process_frame(frame, roi)



        writer.write(annotated)



    writer.release()



if __name__ == "__main__":
    main()