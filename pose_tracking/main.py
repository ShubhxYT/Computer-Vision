import cv2
import mediapipe as mp
import time
from pose_module_tracking import poseDetector

def main():
    cap = cv2.VideoCapture("D:/Codes/Computer Vision/pose_tracking/videos/4.mp4")
    cap.set(3,1280)
    cap.set(4,720)
    ptime = 0
    Detector = poseDetector()
    while True:
        success , img = cap.read()
        img = Detector.findPose(img)
        lmlist = Detector.get_position(img,draw=False)
        cv2.circle(img,(lmlist[14][1],lmlist[14][2]),15,(255,0,255),cv2.FILLED) #right elbow
        cv2.circle(img,(lmlist[13][1],lmlist[13][2]),15,(255,0,255),cv2.FILLED) #left elbow
            
        ctime = time.time()
        fps = 1 / (ctime-ptime)
        ptime = ctime

        cv2.putText(img , str(int(fps)) , (70,50) , cv2.FONT_HERSHEY_PLAIN , 3, (255,0,0),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("finished")
    except Exception as e:
        print(e)
        print("finished")