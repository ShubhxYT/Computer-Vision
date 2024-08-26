import cv2
import time
import os
import HandTrackingModule as htm
wCam, hCam = 1280, 720
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
folderPath = "D:/Codes/Computer Vision/Hand_Tracking/hand_counts"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
    
print(len(overlayList))

pTime = 0
detector = htm.handDetector(detectionCon=0.75,maxHands=1)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img, draw=False)
    # print(lmList)
    #obly works for right hand
    if len(lmList) != 0:
        fingers = []
        
        # Determine if the hand is right or left
        if lmList[17][1] < lmList[5][1]:  # Right hand: base of pinky < base of index finger
            isRightHand = True
        else:  # Left hand: base of pinky > base of index finger
            isRightHand = False
        # Thumb
        if isRightHand:
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # Right hand: thumb tip > thumb base
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:  # Left hand: thumb tip < thumb base
                fingers.append(1)
            else:
                fingers.append(0)
            
        # 4 Fingers
        for id in range(1, 5):
            # checking by cheking the location of x coordinate of the point which is below to the actual
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
        
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]
        
        cv2.rectangle(img, (20, 425), (170, 625), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 575), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break