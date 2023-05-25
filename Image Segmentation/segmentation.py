from ultralytics import YOLO

# load the pretrained model
model = YOLO('yolov8n-seg.pt')

# model.predict(source="Image Segmentation/1.jpeg",save=True, conf=0.25, show=True)

model.predict(source='video_1.mp4', save=True, show=True)
