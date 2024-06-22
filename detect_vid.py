import numpy as np
import cv2 as cv
from ultralytics import YOLO
import random
import torch
from filterpy.kalman import KalmanFilter

def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=4)
    kf.F = np.array([[1, 0, 1, 0],  # state transition matrix
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # measurement function
                     [0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R = np.eye(4) * 10  # measurement uncertainty
    kf.Q = np.eye(4) * 0.1  # process uncertainty
    kf.P *= 1000           # covariance matrix
    kf.x = np.array([0, 0, 0, 0])  # initial state
    return kf

kf = initialize_kalman_filter()

def update_kalman_filter(kf, measurement):
    kf.predict()
    kf.update(measurement)
    return kf.x[:2], kf.x[2:]  # position and velocity

torch.multiprocessing.set_start_method('spawn')  
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : ", device)

# Load a pretrained YOLOv8 model
model = YOLO('runs/detect/train4/weights/best.pt', 'v8')
model.to(device=device)

cap = cv.VideoCapture('vids/rocket_vid3.mp4')
if not cap.isOpened():
    print("Can't play the Video!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    
    cv.imwrite('detected_vid/frame.png', frame)
    results = model.predict(source='detected_vid/frame.png', conf=0.3, save=False)
    
    for result in results:
        boxes = result.boxes
        names = result.names
        if names[0] == 'rocket':  # Class 0 is the rocket.
            for box in boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy()  # Tensorü CPU'ya taşıyıp NumPy dizisine çevir
                midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x1 + x2) / 2, (y1 + y2) / 2])
                position, velocity = update_kalman_filter(kf, midpoint)
                print('Predicted position: ', position, 'Velocity: ', velocity)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (125, 125, 0), 3)
                cv.putText(frame, f'Rocket', (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_COMPLEX, 1, (255, 200, 100), 1)
                future_position = position + velocity
                cv.circle(frame, (int(future_position[0]), int(future_position[1])), 8, (0, 255, 0), -1)

    cv.imshow('detection', frame)
    if cv.waitKey(30) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
