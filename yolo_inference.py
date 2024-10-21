from ultralytics import YOLO
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"


# print("using device: %s" % device)
model = YOLO("yolov8x")

model.predict("input_videos/test.mp4", conf=0.2, save=True)
# result = model.track("input_videos/input_video.mp4", conf=0.2, save=True)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
