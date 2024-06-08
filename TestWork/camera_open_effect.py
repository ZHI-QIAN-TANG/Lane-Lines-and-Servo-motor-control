import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera)
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format = "bgr" , use_video_port = True):
    image = frame.array
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(image,100,200)
    cv2.imshow("frame",image)
    cv2.imshow("gray",gray)
    cv2.imshow("canny",canny)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break