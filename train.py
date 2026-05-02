from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="C:/Users/vaish/Smart_Traffic_Project/dataset/IRUVD/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
