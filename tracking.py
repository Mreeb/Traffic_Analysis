from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
video_path = "Traffic.mp4"
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])
allowed_objects_list = [2,3,5,7]
IDS = list()


while cap.isOpened():
    success, frame = cap.read()
    if success:

        results = model.track(
            frame,
            persist=True,
            classes = allowed_objects_list,
            line_width = 1,
            show_conf = False
            )
        # In Out Functionality Will see if tracking supports that
        # Extacting IDS and there Respective Classes and Store in the Database for Intelegent Retrival
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        IDS.append(track_ids)
        annotated_frame = results[0].plot()
        print(annotated_frame)

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(0, 0, 255),
                thickness=2
                )

        cv2.imshow("YOLO11 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# print(track_history)
# print("\n\n\n\n")
# print("track_ids", track_ids)

# print("\n\n\n\n")
# print("IDS", IDS)
cap.release()
cv2.destroyAllWindows()