import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# import pyautogui

import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
# volume.SetMasterVolumeLevel(0, None)

vol_bar , vol_text = 600,0
while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmlist,bbox = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        x1,y1 = int(lmlist[4][1]), int(lmlist[4][2])
        x2,y2 = int(lmlist[8][1]), int(lmlist[8][2])
        cx,cy = int((x1+x2)/2), int((y1+y2)/2)
        
        cv2.line(img,(x2,y2),(x1, y1),(0,255,0),5)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.dist((x2,y2), (x1, y1))
        #Hand range = 50-300
        vol_value = np.interp(length, [50, 300], [minVol, maxVol])
        vol_text = np.interp(length, [50, 300], [0,100])
        vol_bar = np.interp(length, [50, 300], [600,250])
        volume.SetMasterVolumeLevel(vol_value, None)
        
        print(round(length,2), vol_value)
        
        if length < 50:
            cv2.circle(img, (cx,cy), 15, (0, 255, 0), cv2.FILLED)
            # pyautogui.click()
        elif 50 < length < 125:
            cv2.circle(img, (cx,cy), 15, (0, 255, 255), cv2.FILLED)
            # pyautogui.doubleClick()
        elif 100 < length < 300:
            cv2.circle(img, (cx,cy), 15, (255, 0, 0), cv2.FILLED)
            # pyautogui.rightClick()
    
    cv2.rectangle(img, (50, 250), (85, 600), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 600), (250, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_text)}', (40, 650), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)