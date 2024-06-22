from roboflow import Roboflow
from ultralytics import YOLO
import numpy
import torch

if __name__ == '__main__':
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights


    # Train the model
    results = model.train(data='telesuiveur2-1/data.yaml', epochs=40, imgsz=640) # This pretrained model has been trained for 10 epochs.
    # Note : If you want to get better results. Try to train it for between 20-30 epochs. But I warn you, it will take a lot of time!

