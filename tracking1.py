import json
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolo11n.pt")
video_path = "Traffic11.mp4"
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])
allowed_objects_list = [2, 3, 5, 7]
# Class ID to Name mapping
class_id_to_name = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Dictionary to store ID and their corresponding class
id_class_mapping = {}

while cap.isOpened():
    success, frame = cap.read()
    if success:

        results = model.track(
            frame,
            persist=True,
            classes=allowed_objects_list,
            tracker="bytetrack.yaml"
        )

        # Extracting IDs and their respective classes
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        for track_id, class_id in zip(track_ids, classes):
            if track_id not in id_class_mapping:
                id_class_mapping[track_id] = class_id_to_name.get(class_id, "Unknown")  # Store ID with its class name

        # Annotate frame and track history for visualization
        annotated_frame = results[0].plot(
            conf=False,
            line_width=2,
            kpt_line=True
            )

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

cap.release()
cv2.destroyAllWindows()

# Save IDs and detected classes to JSON without repetition
with open("tracked_ids.json", "w") as file:
    json.dump(id_class_mapping, file, indent=4)
