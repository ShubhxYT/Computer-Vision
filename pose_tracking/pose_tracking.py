import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("D:/Codes/Computer Vision/pose_tracking/videos/3.mp4")
# cap.set(3,1280)
# cap.set(4,720)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

ptime = 0

while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img , results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            # print(id , lm)
            cx , cy = int(lm.x * w) , int(lm.y *h)
            cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
            cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN,1.3, (0, 255, 0), 1) 
        
        
    ctime = time.time()
    fps = 1 / (ctime-ptime)
    ptime = ctime
    
    cv2.putText(img , str(int(fps)) , (70,50) , cv2.FONT_HERSHEY_PLAIN , 3, (255,0,0),3)
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)