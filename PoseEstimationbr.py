import mediapipe as mp
import cv2 as cv
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw =mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

pTime = 0
while True:
    isTrue, frame = cap.read()
    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h,w,c=frame.shape
        print(id,lm)
        cx, cy = int(lm.x*w), int(lm.y*h)
        cv.circle(frame,(cx,cy),6,(155,215,250),cv.FILLED)
    cTime = time.time()
    val=cTime-pTime
    fps=1/val
    pTime=cTime
    cv.putText(frame,str(int(fps)),(70,50),cv.FONT_HERSHEY_PLAIN,4,(255,0,0),2)

    cv.imshow("webcam", frame)
    if cv.waitKey(15) & 0xFF == ord('a'):
        break
