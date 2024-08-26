import cv2
import mediapipe as mp
import time

# cap.set(3,1280)
# cap.set(4,720)

# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils



class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        # self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
        
    def findPose(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(self.results.pose_landmarks)
        
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img , self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
            
        
        return img
    def get_position(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                # print(id , lm)
                cx , cy = int(lm.x * w) , int(lm.y *h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
        if len(lmList) != 0:
            return lmList
            
def main():
    cap = cv2.VideoCapture("D:/Codes/Computer Vision/pose_tracking/videos/3.mp4")
    ptime = 0
    Detector = poseDetector()
    while True:
        success , img = cap.read()
        img = Detector.findPose(img)
        lmlist = Detector.get_position(img,draw=False)
        cv2.circle(img,(lmlist[14][1],lmlist[14][2]),15,(255,0,255),cv2.FILLED) #elbow
        cv2.circle(img,(lmlist[13][1],lmlist[13][2]),15,(255,0,255),cv2.FILLED)
            
        ctime = time.time()
        fps = 1 / (ctime-ptime)
        ptime = ctime

        cv2.putText(img , str(int(fps)) , (70,50) , cv2.FONT_HERSHEY_PLAIN , 3, (255,0,0),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


    
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt as e:
#         print(e)
#         print("finished")
#     except Exception :
#         print("finished")