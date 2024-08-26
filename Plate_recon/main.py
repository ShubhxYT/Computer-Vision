from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import time

# ==== FOR WEBCAM =====
cap = cv2.VideoCapture('D:/Codes/Computer Vision/Plate_recon/licence_plate.mp4')
# cap = cv2.VideoCapture('D:/Codes/Computer Vision/Plate_recon/licence_4k.mp4')
# cap.set(3,1280)
# cap.set(4,720)
pTime = 0

# Load the YOLOv8n model for car detection
car_model = YOLO('D:/Codes/Computer Vision/Yolo-Weights/yolov8n.pt')

# Load the license plate recognition model
plate_model = YOLO('D:/Codes/Computer Vision/Yolo-Weights/plate_recon.pt')

# Create a tracker for license plates
plate_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Create a tracker for cars
car_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Detect license plates
    license_plates = plate_model(img)
    
    # Detect cars
    cars = car_model(img)
    
    # Update license plate tracker
    plate_detections = np.empty((0, 5))
    for r in license_plates:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            currentArray = np.array([x1, y1, x2, y2, conf])
            plate_detections = np.vstack((plate_detections, currentArray))
    plate_results = plate_tracker.update(plate_detections)
    
    # Update car tracker
    car_detections = np.empty((0, 5))
    for r in cars:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            currentArray = np.array([x1, y1, x2, y2, conf])
            car_detections = np.vstack((car_detections, currentArray))
    car_results = car_tracker.update(car_detections)
    
    # Process license plate and car results
    for car_result in car_results:
        # print(f"\n\n{car_result}\n\n")
        x1, y1, x2, y2, car_id = car_result
        x1, y1, x2, y2, car_id = int(x1), int(y1), int(x2), int(y2), int(car_id)
        # if plate_id == car_id:
        # Draw the car bounding box and ID
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=2, colorR=(0, 255, 0))
        # cvzone.putTextRect(img, f'Car: {car_id}', (max(0, x1), max(30, y1)), scale=4, thickness=2, offset=3)
        
        for plate_result in plate_results:
            x1, y1, x2, y2, plate_id = plate_result
            x1, y1, x2, y2, plate_id = int(x1), int(y1), int(x2), int(y2), int(plate_id)

            # Extract the license plate region
            license_plate_img = img[y1:y2, x1:x2]

            # Convert the image to grayscale
            gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

            # Apply some preprocessing (optional)
            # gray_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)
            _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
            # Resize the license plate image (increase size)
            scale_factor = 3  # Adjust this factor to increase/decrease size
            new_width = int((x2 - x1) * scale_factor)
            new_height = int((y2 - y1) * scale_factor)
            # resized_binary_plate = cv2.resize(binary_plate, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_licence_plate = cv2.resize(license_plate_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Calculate the position to place the resized image above the original plate
            new_y1 = max(0, y1 - new_height)  # Ensure we don't go above the image
            new_x1 = x1-90  # Align with the original x1 position
    
            # Overlay the resized image onto the original image
            # Ensure we don't go out of bounds
            if new_y1 >= 0 and new_x1 + new_width <= img.shape[1]:
                # img[new_y1:new_y1 + new_height, new_x1:new_x1 + new_width] = cv2.cvtColor(resized_binary_plate, cv2.COLOR_GRAY2BGR)
                img[new_y1:new_y1 + new_height, new_x1:new_x1 + new_width] = resized_licence_plate

            # Draw the license plate bounding box and ID
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=2, colorR=(255, 0, 0))
            # cvzone.putTextRect(img, f'Plate: {car_id}', (max(0, x1), max(50, y1)), scale=4, thickness=2, offset=3)
            # print("\n\n\n")
            # Match the license plate ID with car IDs
                
    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    # Display the resulting image
    resized_frame = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break