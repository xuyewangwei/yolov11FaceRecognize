from  ultralytics import YOLO
from ultralytics import settings
import cv2

settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})
settings.update({"runs_dir": "./run/"})

model = YOLO("./best.onnx")
pic_path = "/root/autodl-tmp/yolov11/datasets/Face-Detection/test/images/Movie-on-2-18-25-at-8_25-PM_mov-0184_jpg.rf.8228aff02719eff3dadda956e01a28b9.jpg"
result = model(pic_path)
result[0].save("output.jpg")