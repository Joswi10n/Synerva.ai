#!/usr/bin/env python3
# cctv_yolo.py — YOLOv5 + DeepSORT + InsightFace (RetinaFace + ArcFace)
# Plays back at the original FPS and assigns a single persistent name per track.

import argparse, os, cv2, torch, pickle, numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis   # RetinaFace detector + ArcFace recogniser

# ─────────────────────────────────────────────────────────────────────────────
# Persistent face database ────────────────────────────────────────────────────
DB_PATH = "known_faces.pkl"
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        known_faces = pickle.load(f)      # {name: 512-D embedding}
else:
    known_faces = {}

# InsightFace initialisation (CPUExecutionProvider on WSL)
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))   # RetinaFace detector size

# ─────────────────────────────────────────────────────────────────────────────
def get_embedding(crop):
    """Return 512-D ArcFace embedding or None if no face."""
    faces = face_app.get(crop)
    return faces[0].embedding if faces else None

def recognize_face(crop, threshold=1.2):
    """Return known name within threshold, otherwise None."""
    emb = get_embedding(crop)
    if emb is None or not known_faces:
        return None
    dists = {n: np.linalg.norm(emb - ref) for n, ref in known_faces.items()}
    name, dist = min(dists.items(), key=lambda x: x[1])
    return name if dist < threshold else None

def enroll_face(crop, name):
    emb = get_embedding(crop)
    if emb is None:
        return False
    known_faces[name] = emb
    with open(DB_PATH, "wb") as f:
        pickle.dump(known_faces, f)
    return True

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLOv5 + DeepSORT + InsightFace")
    parser.add_argument("-s", "--source", default="0",
                        help="Camera index or video/RTSP path")
    parser.add_argument("-c", "--conf", type=float, default=0.4,
                        help="YOLOv5 confidence threshold")
    parser.add_argument("--auto-enroll", action="store_true",
                        help="Auto-enrol unknown faces as person_N")
    args = parser.parse_args()

    # Resolve source (int for webcam, string for file/URL)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))
    print(f"[INFO] Source={source}   conf={args.conf}   auto_enroll={args.auto_enroll}")

    # YOLOv5 (person-only)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = args.conf
    model.classes = [0]          # class 0 = person

    # DeepSORT tracker
    tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2)
    track_to_name = {}           # {track_id: label}

    # Video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000.0 / fps))
    print(f"[INFO] FPS={fps:.2f}  display-delay={delay} ms")

    frame_h = frame_w = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_h is None:
            frame_h, frame_w = frame.shape[:2]

        # YOLO detections
        dets = []
        for *xyxy, conf, _ in model(frame).xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), 'person'))

        # DeepSORT update
        tracks = tracker.update_tracks(dets, frame=frame)
        active = {t.track_id for t in tracks if t.is_confirmed()}
        track_to_name = {tid: lab for tid, lab in track_to_name.items() if tid in active}

        # Draw tracks
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            tid       = trk.track_id
            x1, y1, x2, y2 = map(int, trk.to_ltrb())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            crop = frame[y1:y2, x1:x2]

            # ── Guard against empty crops ────────────────────────────────────
            h, w = crop.shape[:2]
            if h == 0 or w == 0:
                continue                      # skip this track safely
            # Optional up-sampling for tiny face boxes
            if min(h, w) < 50:
                crop = cv2.resize(crop, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
            # ─────────────────────────────────────────────────────────────────

            # Assign / look up label
            if tid not in track_to_name:
                name = recognize_face(crop)
                if name is None and args.auto_enroll:
                    name = f"person_{len(known_faces) + 1}"
                    enroll_face(crop, name)
                track_to_name[tid] = name or f"ID {tid}"

            label = track_to_name[tid]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Tracker", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
