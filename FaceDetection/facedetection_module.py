import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture(0)
class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        
    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        # print(results)
        bboxs = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                # self.mpDraw.draw_detection(img, detection) #to draw the points
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img = self.fancDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',(bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img,bboxs
    
    def fancDraw(self, img, bbox,l=30,t=7):
        x,y,w,h = bbox
        x1 , y1 = x+w , y+h
        cv2.rectangle(img, bbox, (0, 255, 0), 2)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    cap = cv2.VideoCapture("D:/Codes/Computer Vision/FaceDetection/videos/6.mp4")
    # cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img,bboxs = detector.findFaces(img)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
if __name__ == "__main__":
    main()