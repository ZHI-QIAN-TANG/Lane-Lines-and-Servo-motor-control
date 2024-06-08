"""include library"""
import cv2
import apriltag
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import argparse
import numpy as up
camera = PiCamera()#open camara
camera.resolution = (640,480)#settings windows long and width
camera.framerate = 64 #settings framerate 32
rawCapture = PiRGBArray(camera) #settings original picture RGBArray
time.sleep(0.1)#open camara time
for frame in camera.capture_continuous(rawCapture, format = "bgr" , use_video_port = True):#settings picture is bgr and show picture
    image = frame.array #read frame
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) ##settings gray
    detector = apriltag.Detector()#detect apriltag
    result = detector.detect(gray)#result gray picture
    for apriltagsuccess in result:#confirm apriltag success
        tagname = apriltagsuccess.tag_family.decode("utf-8")#shoe apriltag name
        print(tagname)
        (pt1,pt2,pt3,pt4) = apriltagsuccess.corners #search apriltag corners
        pt1 = (int(pt1[0]),int(pt1[1]))
        pt2 = (int(pt2[0]),int(pt2[1]))
        pt3 = (int(pt3[0]),int(pt3[1]))
        pt4 = (int(pt4[0]),int(pt4[1]))
        cv2.line(image,pt3,pt4,(0,255,0),2)#painting line
        cv2.line(image,pt1,pt2,(0,255,0),2)#painting line
        cv2.line(image,pt2,pt3,(0,255,0),2)#painting line
        cv2.line(image,pt4,pt1,(0,255,0),2)#painting line
    cv2.imshow("frame",image) #show image
    key = cv2.waitKey(1) & 0xFF #setting key
    rawCapture.truncate(0) #clear rawcapture
    if key == ord("q"):
        break