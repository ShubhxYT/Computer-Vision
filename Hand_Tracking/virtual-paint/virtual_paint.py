import cv2
import time
import os
import numpy as np
import math

import sys
sys.path.insert(0, 'D:/Codes/Computer Vision/Hand_Tracking/')
import HandTrackingModule as htm

RECTANGLE_COLOUR = (255, 0, 255)
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 50
TAP_ERASE = 5.0
FOLDER_PATH = "D:/Codes/Computer Vision/Hand_Tracking/hand-paint/menu"

myList = os.listdir(FOLDER_PATH)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{FOLDER_PATH}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#starting as red as default
draw_color = (0, 0, 255)
xp , yp = 0, 0
tap = 0

detector = htm.handDetector(detectionCon=0.5, maxHands=1)
header = overlayList[0]
img_canvas = np.zeros((hCam,wCam,3),np.uint8)

while True :
    success, img = cap.read()
    #flipping the image so that it is not mirrored
    img = cv2.flip(img, 1)
    #adding the header bourd
    img[0:97,0:1280] = header
    
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img, draw=False)
    # print(lmList)
    
    if len(lmList) != 0:
        
        x1,y1 = lmList[8][1],lmList[8][2]
        x2,y2 = lmList[12][1],lmList[12][2]
        
        OneFingerUp = False
        TwoFingerUp = False
        
        if y1 < lmList[8-2][2]:
            OneFingerUp=True
            if y2 < lmList[12-2][2]:
                OneFingerUp = False
                TwoFingerUp = True
                
        #taking out the length of between the fingers tips
        length =math.dist((int(lmList[8][1]),int(lmList[8][2])), (int(lmList[4][1]), int(lmList[4][2])))
        
        if length < 30:
            tap += 1
            #making sure these both don't run
            OneFingerUp=False
            TwoFingerUp=False
            cv2.circle(img, (x1, y1), 30, (0, 255, 0), cv2.FILLED)
            if tap == 1 :
                start_time = time.time()
            if tap >= 2:
                time_left = TAP_ERASE-(np.round(time.time()-start_time,1))
                cv2.putText(img, f"{time_left:.1f}", (x1-17 , y1+7), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                if time_left <= 0:
                    tap = 0
                    xp , yp = 0, 0
                    #clearing the canvas painting
                    cv2.rectangle(img_canvas, (0, 0), (wCam, hCam), (0,0,0), cv2.FILLED)
                
              
        if OneFingerUp:
            tap = 0 #resseting tap_counter timer
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if draw_color == (255, 255, 255):
                cv2.circle(img, (x1, y1), 33, draw_color, cv2.FILLED)
                cv2.line(img_canvas, (xp, yp), (x1, y1), (0,0,0), ERASER_THICKNESS)
            else:
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, BRUSH_THICKNESS)
            #updating the previous points
            xp, yp = x1, y1
                       
        if TwoFingerUp:
            tap = 0
            xp, yp = 0,0
            cv2.rectangle(img, (x1, y1), (x2, y2), RECTANGLE_COLOUR, cv2.FILLED)
            
            if 171<x1<256 and 171<x2<256 and y1<125 and y2<125:
                header = overlayList[0]
                draw_color = (0, 0, 255) #red
                
            if 470<x1<555 and 470<x2<555 and y1<125 and y2<125:
                header = overlayList[1]
                draw_color = (255, 0, 0) #blue
                
            if 761<x1<836 and 761<x2<836 and y1<125 and y2<125:
                header = overlayList[2]
                draw_color = (10, 10, 10) #black
            
            if 1036<x1<1153 and 1036<x2<1153 and y1<125 and y2<125:
                header = overlayList[3]
                draw_color = (255, 255, 255) #white
            
    #making the grey canvas for all the colours         
    imgGray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    #converting their colours to black and white
    _, imgInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    #inserting the image overlay
    img = cv2.bitwise_and(img,imgInv)
    #interting back all the colours
    img = cv2.bitwise_or(img,img_canvas)
    
    # for merging both images or overlaping  
    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)


