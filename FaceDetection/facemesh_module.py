import cv2
import mediapipe as mp
import time

face_connections = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
]

class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    # self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,self.drawSpec, self.drawSpec)
                    self.mpDraw.draw_landmarks(img, faceLms, None, self.drawSpec, self.drawSpec)
                    for connection in face_connections:
                        x1, y1 = int(faceLms.landmark[connection[0]].x * img.shape[1]), int(faceLms.landmark[connection[0]].y * img.shape[0])
                        x2, y2 = int(faceLms.landmark[connection[1]].x * img.shape[1]), int(faceLms.landmark[connection[1]].y * img.shape[0])
                        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    if draw:
                        #for numbers for each point
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,0.7, (0, 255, 0), 1) 
                    #print(id,x,y)
                    face.append([x,y])
                    
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture("D:/Codes/Computer Vision/FaceDetection/videos/2.mp4")
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img,True)
        # if len(faces)!= 0:
        #     print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()