from math import isnan
import cv2
from cv2.typing import MatLike

def drawLinesByMean(frame: MatLike, lines: MatLike):
    
    leftx1 = []
    leftx2 = []
    lefty1 = []
    lefty2 = []
    
    rightx1 = []
    rightx2 = []
    righty1 = []
    righty2 = []
    
    (_, height, _) = frame.shape
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = ((height - y1) - (height - y2)) / (x1 - x2)
        
        if slope > 0:
            if slope > 0.4 and slope < 1:
                leftx1.append(x1)
                leftx2.append(x2)
                lefty1.append(y1)
                lefty2.append(y2)
        else:
            if slope < -0.4 and slope > -1:
                rightx1.append(x1)
                rightx2.append(x2)
                righty1.append(y1)
                righty2.append(y2)
    
    if len(leftx1) > 0 and len(leftx2) > 0 and len(lefty1) > 0 and len(lefty2) > 0:
        mlx1 = sum(leftx1) / len(leftx1)
        mlx2 = sum(leftx2) / len(leftx2)
        mly1 = sum(lefty1) / len(lefty1)
        mly2 = sum(lefty2) / len(lefty2)
        lslope = (mly1 - mly2) / (mlx1 - mlx2)
        lvalue = mly1 - lslope * mlx1
        
        if mlx1 != mlx2 and mly1 != mly2:
            ly1 = round(height * (5 / 12))
            lx1 = round((ly1 - lvalue) / lslope)
            ly2 = height
            lx2 = round((ly2 - lvalue) / lslope)
            
            cv2.line(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 3)
    
    if len(rightx1) > 0 and len(rightx1) > 0 and len(righty1) > 0 and len(righty2) > 0:
        mrx1 = sum(rightx1) / len(rightx1)
        mrx2 = sum(rightx2) / len(rightx2)
        mry1 = sum(righty1) / len(righty1)
        mry2 = sum(righty2) / len(righty2)
        
        if mrx1 != mrx2 and mry1 != mry2:
            rslope = (mry1 - mry2) / (mrx1 - mrx2)
            rvalue = mry1 - rslope * mrx1

            ry1 = round(height * (5 / 12))
            rx1 = round((ry1 - rvalue) / rslope)
            ry2 = height
            rx2 = round((ry2 - rvalue) / rslope)
            
            cv2.line(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
            
import numpy as np
from scipy.stats import linregress

def pointRegression(points: np.ndarray[tuple[int, int]]):
    nlist = points.tolist()
    xs = list(map(lambda x: x[0], nlist))
    ys = list(map(lambda x: x[1], nlist))
    
    result = linregress(xs, ys)
    
    return result.slope