from ultralytics import YOLO
import json

MODELS = "models.json"
FILE = "Traffic_flow.mp4"


# 2:Car, 3:motorcycle, 5:bus, 7:truck
allowed_objects_list = [2,3,5,7]


with open(MODELS,  "r") as file:
    models  = json.load(file)

model = YOLO(models["Detection"]["Model1"])

# model = YOLO("yolo11n.pt")
# print(model.names)

results = model(
    FILE,
    save=True,
    show=True,
    classes=allowed_objects_list,
    line_width = 1,
    show_conf = False
    )
