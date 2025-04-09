from  ultralytics import YOLO
from ultralytics import settings

settings.update({"weights_dir": "./models/"})
settings.update({"datasets_dir": "./datasets/"})
settings.update({"runs_dir": "./run/"})
model = YOLO("./models/yolo11n.pt")

results = model.train(data='./datasets/Face-Detection.yaml',epochs=100) # 这里调用的现有数据库

#model.export(format="onnx")
model.export(format="onnx")