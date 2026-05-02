from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="ADD THE FILE LOCATION",
    epochs=50,
    imgsz=640,
    batch=8
)
