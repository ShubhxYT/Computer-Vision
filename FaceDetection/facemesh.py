import cv2
import mediapipe as mp
import time

# Initialize video capture
# cap = cv2.VideoCapture("D:/Codes/Computer Vision/FaceDetection/videos/2.mp4")
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
pTime = 0

# Initialize MediaPipe Face Mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Define face connections
face_connections = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS , drawSpec, drawSpec)
            mpDraw.draw_landmarks(img, faceLms, None, drawSpec, drawSpec)
            for connection in face_connections:
                x1, y1 = int(faceLms.landmark[connection[0]].x * img.shape[1]), int(faceLms.landmark[connection[0]].y * img.shape[0])
                x2, y2 = int(faceLms.landmark[connection[1]].x * img.shape[1]), int(faceLms.landmark[connection[1]].y * img.shape[0])
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            for id ,ml in enumerate(faceLms.landmark):
                ih , iw , ic = img.shape
                x, y = int(ml.x * iw), int(ml.y * ih)
                # print(id, x, y)
                # cv2.circle(img, (x, y), 1, (0, 255, 0), 1)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()