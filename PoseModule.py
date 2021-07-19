import mediapipe as mp
import cv2 as cv
import time
import math

class poseDetector():

    def __init__(self, mode=False, upbody=False, smooth = True, detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.upbody=upbody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon= trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.upbody,self.smooth,self.detectionCon,self.trackCon)
        self.mpDraw =mp.solutions.drawing_utils

    def findPose(self,img, draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmlist=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape

                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                if draw:
                     cv.circle(img,(cx,cy),6,(0,0,250),cv.FILLED)
        return self.lmlist

    def findAngle(self,img,p1,p2,p3,draw=True):
    #Getting the Landmarks
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]
    #Finding The Angle

        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        if angle<0:
            angle+=360


        if draw:
            cv.circle(img,(x1,y1),10,(0,0,255),-1)
            cv.circle(img,(x1,y1),20,(0,0,255),2)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), -1)
            cv.circle(img, (x2, y2), 20, (0, 0, 255), 2)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), -1)
            cv.circle(img, (x3, y3), 20, (0, 0, 255), 2)
            cv.line(img,(x1,y1),(x2,y2),(255,255,255),4)
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 4)
            cv.putText(img,str(int(angle)),(x2+35,y2),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
            return angle

def main():
    cap = cv.VideoCapture("pose vi.mp4")
    detector = poseDetector()

    pTime = 0
    while True:
        isTrue, frame = cap.read()
        detector.findPose(frame)
        landmarklist=detector.findPosition(frame, draw=False)

        cTime = time.time()
        val = cTime - pTime
        fps = 1 / val
        pTime = cTime
        cv.putText(frame, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 2)

        cv.imshow("webcam", frame)
        if cv.waitKey(15) & 0xFF == ord('a'):
            break

if(__name__)=='__main__':
    main()
