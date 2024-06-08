import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from win32api import GetSystemMetrics
from track import cannyTrack, maskTrack
from util import drawLinesByMean


cap = cv2.VideoCapture("德國無限速高速公路-快車道自動讓行.mp4")  # 將影片導入

# 讀取鏡頭方式:
# cap = cv2.VideoCapture(0)#0為筆記型電腦專用鏡頭編號
# cv2.createTrackbar(控制條名稱,此控制條的位置,控制條的初始值,控制條的最大值,控制條被改變時所需要動的函式)
screen: dict["width": int, "height": int] = {"width": GetSystemMetrics(0), "height": GetSystemMetrics(1)}
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def window(name: str, x: int = 0, y: int = 0, width: int = width, height: int = height):
    cv2.namedWindow(name)
    cv2.moveWindow(name, x, y)    
    cv2.resizeWindow(name, width, height)


win = "video"
window(win)

maskTrackName = "masktrack"
window(maskTrackName, width=500, height=250)
maskTracks = maskTrack(maskTrackName)

cannyTrackName = "canny"
window(cannyTrackName, width=500, height=100)
cannyTracks = cannyTrack(cannyTrackName)

while True:
    ret, frame = (
        cap.read()
    )  # 讀取影響影片前項ret=前一禎(為布林值)frame=後一禎(為布林值)
    if ret == True:
        
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
        low_white = np.array([maskTracks["lowh"](), maskTracks["lows"](), maskTracks["lowv"]()])
        high_white = np.array([maskTracks["highh"](), maskTracks["highs"](), maskTracks["highv"]()])
        mask = cv2.inRange(hsv, low_white, high_white)

        canny = cv2.Canny(mask, cannyTracks["low"](), cannyTracks["high"]())

        roi_mask = np.zeros(canny.shape, dtype=np.uint8)
        roi = np.array([[width * (2 / 5 - 1 / 10), height * (1 / 2)],
                        [width * (1 / 2), height * (1 / 2)],
                        [width * (1 / 2 + 1 / 3), height], 
                        [0, height]],np.int32)

        cv2.fillConvexPoly(roi_mask, roi, 255)
        roi_canny = cv2.bitwise_and(canny, roi_mask)
        
        lines = cv2.HoughLinesP(
            roi_canny, 1, np.pi / 180, 25, maxLineGap=10, minLineLength=20
        )
        
        
        if(lines is not None):
            drawLinesByMean(frame, lines)
            
        cv2.imshow(win, frame)      
        # cv2.bitwise_and(要改變的圖像,要改變的圖像,過濾)
        # 秀出圖片(#每個影片皆是用圖片一層一層顯示)
   
        # cv2.imshow("video4", face)
    if ret == False:
        break  # 播完就結束
    if cv2.waitKey(20) == ord(
        "q"
    ):  # waitKer(1)為顯示完圖片後所需要等待的毫秒數，數字越大影片速度越慢，ord('q')代表按下此鍵後會直接結束影片
        break
