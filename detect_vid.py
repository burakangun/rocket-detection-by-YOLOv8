import numpy as np
import cv2 as cv
from ultralytics import YOLO
import random
import torch


torch.multiprocessing.set_start_method('spawn')  

device: str = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : ",device)

#loading a pretrained YOLOv8n model
model = YOLO('runs/detect/train4/weights/best.pt', 'v8')
model.to(device=device)

# loading the video
cap = cv.VideoCapture('rocket_vid3.mp4') #You can change the vid if you want. Moreover, you can test the model by other videos you find in the youtube.

# check if the cap is not opened
if not cap.isOpened():
    print("Can't play the Video!")
    exit()

while True:
    #capture frame-by-frame
    ret, frame = cap.read()       
    #if frame is read correctly ret will be true

    if not ret:
        print("Can't receive frame")
        break
    
    cv.imwrite('detected_vid/frame.png',frame)

    results = model.predict(source='detected_vid/frame.png', conf=0.22,save=False) # The best conf value is 0.22 for now!    
    
    
    for result in results:
        boxes = result.boxes
        names = result.names
        if names[0] == 'rocket':  # Class 0 is the rocket.                   
            
            if len(boxes) > 0:
                x1, y1, x2, y2= boxes.xyxy[0]
                print('Coords : ',x1,y1,x2,y2)  
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (125, 125, 0), 3)
                font = cv.FONT_HERSHEY_COMPLEX
                cv.putText(frame, 'rocket', (int(x1), int(y1-5)), cv.FONT_HERSHEY_COMPLEX, 1, (255, 200, 100), 1)
    
    cv.imshow('detection',frame)
    
    if cv.waitKey(30) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()