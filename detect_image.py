import numpy as np
import cv2 as cv
from ultralytics import YOLO
import random

#loading a pretrained YOLOv8n model
model = YOLO('runs/detect/train4/weights/best.pt', 'v8')

# Load the Image
image_path = 'images/rocket.jpg'  # The path to the image
image = cv.imread(image_path)

# Detect the rockets
results = model(image)

# Draw bounding boxes around the detected rockets"

for result in results:
    boxes = result.boxes
    names = result.names 
    print('Box : ',boxes)
    print('Names : ',names)
    
    if names[0] == 'rocket':  # 0. sınıf roket olduğunu varsayalım                
        
        x1, y1, x2, y2= boxes.xyxy[0]
        print('Koordinatlar : ',x1,y1,x2,y2)  
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        font = cv.FONT_HERSHEY_COMPLEX
        cv.putText(image, 'rocket', (int(x1), int(y1-5)), cv.FONT_HERSHEY_COMPLEX, 1, (255, 200, 100), 1)
    

# Sonuçları görüntüleq
cv.imshow('Detected Rockets', image)
cv.waitKey(0)
cv.destroyAllWindows()