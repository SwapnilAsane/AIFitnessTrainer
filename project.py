import PoseModule as pm
import cv2 as cv
import numpy as np
import time

#Import necessary libraries
from flask import Flask, render_template, Response

#Initialize the Flask app
app = Flask(__name__)
cap = cv.VideoCapture(0)

def gen_frames():

            # img = cv.imread("Resources/bicep.jpg")
            detector = pm.poseDetector()
            dir = 0
            count = 0
            while True:
                success, img = cap.read()
                img = cv.resize(img, (1280, 720))
                # img = cv.imread("Resources/bicep.jpg")
                img = detector.findPose(img, False)
                lmlist = detector.findPosition(img, False)
                if len(lmlist) != 0:
                    # LeftArm
                    angle = detector.findAngle(img, 11, 13, 15)
                    # Right Arm
                    per = np.interp(angle, (210, 310), (0, 100))
                    bar = np.interp(angle, (220, 310), (650, 100))
                    color = (165, 166, 149)
                    # Check for the curls
                    if per == 100:
                        color = (66, 245, 179)
                        if dir == 0:
                            count += 0.5
                            dir = 1
                    if per == 0:
                        color = (66, 245, 179)
                        if dir == 1:
                            count += 0.5
                            dir = 0

                    # Draw Bar

                    cv.rectangle(img, (1100, 100), (1175, 650), color, 3)
                    cv.rectangle(img, (1100, int(bar)), (1175, 650), color, cv.FILLED)
                    cv.putText(img, f'{int(per)}%', (1100, 75), cv.FONT_HERSHEY_PLAIN, 4, color, 4)

                    # curl count

                    cv.rectangle(img, (0, 0), (150, 150), (255, 255, 255), -1)
                cv.putText(img, str(int(count)), (50, 100), cv.FONT_HERSHEY_PLAIN, 7, (60, 76, 231), 3)


                if cv.waitKey(15) & 0xFF == ord('a'):
                    break

                ret, buffer = cv.imencode('.jpg', img)
                img = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

