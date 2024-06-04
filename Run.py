import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#讀取影片方式:
def empty():
    pass
cap = cv2.VideoCapture('德國無限速高速公路-快車道自動讓行.mp4')#將影片導入
#讀取鏡頭方式:
#cap = cv2.VideoCapture(0)#0為筆記型電腦專用鏡頭編號
#cv2.createTrackbar(控制條名稱,此控制條的位置,控制條的初始值,控制條的最大值,控制條被改變時所需要動的函式)
while True:
    ret,frame= cap.read()#讀取影響影片前項ret=前一禎(為布林值)frame=後一禎(為布林值) 
    if ret == True:
        face = frame[500:1080, 100:1920]
        blur = cv2.GaussianBlur(face,(3,3),0)
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        low_weite = np.array([19,16,198])
        high_weite = np.array([33,61,255])
        mask = cv2.inRange(hsv,low_weite,high_weite)
        canny = cv2.Canny(mask,50,80)
        lines = cv2.HoughLinesP(canny,1,np.pi/180,25,maxLineGap=180,minLineLength=20)
        if lines is None:
            pass
        else:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(face,(x1,y1),(x2,y2),(255,0,0),3)
    #cv2.bitwise_and(要改變的圖像,要改變的圖像,過濾)
        cv2.imshow('video',frame)#秀出圖片(#每個影片皆是用圖片一層一層顯示)
        cv2.imshow('video2',canny)
        cv2.imshow('video4',face)
    if ret == False:
        break#播完就結束
    if  cv2.waitKey(20) == ord('q'):#waitKer(1)為顯示完圖片後所需要等待的毫秒數，數字越大影片速度越慢，ord('q')代表按下此鍵後會直接結束影片
       break