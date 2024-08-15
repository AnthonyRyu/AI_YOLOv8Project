from ultralytics import YOLO

model = YOLO("best_butterfly_seg.pt")

model.predict(source="butterfly.jpg", show=True, save=True, show_labels=True, show_conf=True, conf=0.5, line_width=2)