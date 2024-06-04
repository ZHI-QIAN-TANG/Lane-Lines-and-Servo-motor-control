import cv2
from cv2.typing import MatLike
import pandas as pd
import numpy as np
from win32api import GetSystemMetrics

# 讀取影片方式:
def empty():
    pass

cap = cv2.VideoCapture("lane_vid2.mp4")  # 將影片導入

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

def region_of_interest(img):
    mask = np.zeros_like(img)
    imshape=img.shape
    vertices = np.array([[(150,imshape[0]),(590, 440), (680, 440), (imshape[1]-20,imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


while True:
    ret, frame = (
        cap.read()
    )  # 讀取影響影片前項ret=前一禎(為布林值)frame=後一禎(為布林值)
    if ret == True:
        blur = cv2.medianBlur(frame, 5)
        
        # hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        low_white = np.array([18, 0, 46])
        high_white = np.array([34, 255, 255])
        mask = cv2.inRange(hsv, low_white, high_white)
        v = mask[:: 1]
        ret, thresh = cv2.threshold(v, 50, 80, cv2.THRESH_BINARY)
        canny = cv2.Canny(thresh, 100, 200).astype(np.float32)
        sobel = cv2.Sobel(thresh, cv2.CV_64F, 0, 1, ksize=7).astype(np.float32)
        weighted = cv2.addWeighted(sobel, 0.7, canny, 0.3, 0)
        combine = cv2.bitwise_and(weighted, sobel)
        roi = region_of_interest(combine)

        src = np.float32([[480, 500], [800, 500], [roi.shape[1]-50, roi.shape[0]],  [150, roi.shape[0]]])
        
        line_dst_offset = 200
        dst = np.float32([
                [src[3][0] + line_dst_offset, 0],
                [src[2][0] - line_dst_offset, 0],
                [src[2][0] - line_dst_offset, src[2][1]],
                [src[3][0] + line_dst_offset, src[3][1]]
            ])
        
        p = cv2.getPerspectiveTransform(src, dst)
        wp: MatLike = cv2.warpPerspective(roi, p, roi.shape[0: 2][::-1], cv2.INTER_LINEAR)
        
        # histogram
        his: np.ndarray = np.sum(wp[mask.shape[0] // 2:, :], axis=0, dtype=np.uint32)
        mid = int(his.shape[0] / 2)
        left = np.argmax(his[: mid])
        right = np.argmax(his[mid: ]) + mid

        # sliding window
        ty = 0
        by = 712
        lx = []
        rx = []
        mk = np.zeros(wp.shape, dtype=np.float32)
        
        while by > ty:
            # left threshold
            l1 = int(left - 50)
            l2 = int(left + 50)
            img = np.uint8(wp[by - 40: by, l1: l2])
            coutrs, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for coutr in coutrs:
                m = cv2.moments(coutr)
                if m["m00"] != 0:
                    cx = m["m10"] // m["m00"]
                    cy = m["m01"] // m["m00"]
                    lx.append(l1 + cx)
                    left = left - 50 + cx
            
            cv2.rectangle(mk, (l1, int(by)), (l2, int(by - 40)), (255, 255, 255), 2)
            
            # right threshold
            r1 = int(right - 50)
            r2 = int(right + 50)
            
            img = np.uint8(wp[by - 40: by, r1: r2])
            coutrs, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for coutr in coutrs:
                m = cv2.moments(coutr)
                if m["m00"] != 0:
                    cx = m["m10"] // m["m00"]
                    cy = m["m01"] // m["m00"]
                    rx.append(r1 + cx)
                    reft = right - 50 + cx
                    
            cv2.rectangle(mk, (r1, int(by)), (r2, int(by - 40)), (255, 255, 255), 2)
            
            by -= 40
        
        cv2.imshow("w", v)
        # cv2.imshow("win", frame[height // 2:, 0: width])
        # cv2.resizeWindow("win", screen["width"], screen["height"] // 2)
        # cv2.imshow("win1", wp)
        # cv2.resizeWindow("win1", screen["width"], screen["height"] // 2)
        # cv2.moveWindow("win1", 0, screen["height"] // 2)
        # cv2.imshow("win2", mk)
        # cv2.resizeWindow("win2", screen["width"], screen["height"] // 2)
        # cv2.moveWindow("win2", 0, screen["height"] // 2)
    if ret == False:
        break  # 播完就結束
    if cv2.waitKey(20) == ord(
        "q"
    ):  # waitKer(1)為顯示完圖片後所需要等待的毫秒數，數字越大影片速度越慢，ord('q')代表按下此鍵後會直接結束影片
        break
