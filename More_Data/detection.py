from ultralytics import YOLO

 # Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=100, workers=0, device=0)  # train the model


