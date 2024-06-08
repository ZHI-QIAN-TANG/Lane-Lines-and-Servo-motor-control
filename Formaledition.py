#用於跑電腦模擬
import numpy as np # 導入NumPy庫
import cv2 # 導入OpenCV庫
import time # 導入時間庫
from scipy.stats import linregress

out_examples = 0 # 初始化變量out_examples為0
MOV_AVG_LENGTH = 5 # 設定移動平均的長度為5

# 定義顏色處理函數
def ProcessImage(inpImage):
    # 設定白色的下界和上界
    lowerWhite = np.array([0, 160, 10])
    upperWhite = np.array([255, 255, 255])
    
    # 對圖像進行閾值處理，提取白色部分
    mask = cv2.inRange(inpImage, lowerWhite, upperWhite)
    hlsResult = cv2.bitwise_and(inpImage, inpImage, mask=mask)
    
    # 將結果轉換為灰度圖像
    gray = cv2.cvtColor(hlsResult, cv2.COLOR_BGR2GRAY)
    
    # 對灰度圖像進行自適應閾值處理
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 對二值圖像進行高斯模糊
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    
    # 使用Canny邊緣檢測
    canny = cv2.Canny(blur, 50, 150)
    
    # 進行膨脹
    kernel = np.ones((9, 9), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    
    # 顯示擴張後的圖像（在實際使用時可以選擇移除）
    cv2.imshow('Edges', dilate)
    return dilate


def RegionOfInterest(img, vertices=None):
    mask = np.zeros_like(img)  # 創建與輸入圖像相同大小的黑色遮罩

    if vertices is None:
        imshape = img.shape
        # 預設多邊形頂點
        vertices = np.array([[(0, imshape[0]), (390, 440), (880, 440), (imshape[1], imshape[0])]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, 255)  # 填充多邊形
    masked_image = cv2.bitwise_and(img, mask)  # 應用遮罩

    # 顯示遮罩後的圖像（在實際使用時可以選擇移除）
    cv2.imshow('Masked Image', masked_image)

    return masked_image  # 返回遮罩後的圖像

# 示例使用
# inpImage = cv2.imread('path_to_your_image.jpg')
# vertices = np.array([[(0, inpImage.shape[0]), (390, 440), (880, 440), (inpImage.shape[1], inpImage.shape[0])]], dtype=np.int32)
# roi_image = RegionOfInterest(inpImage, vertices)


def warp(img, src_points, dst_points, size=None):
    # 檢查並轉換源點和目標點為浮點數類型
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    
    # 獲取圖像尺寸
    img_size = (img.shape[1], img.shape[0]) if size is None else size
    
    # 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(src, dst)
    
    # 進行透視變換
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    # 創建一個黑色圖像並將透視變換後的圖像置於中央
    centered_warped = np.zeros_like(img)
    x_offset = (centered_warped.shape[1] - img_size[0]) // 2
    y_offset = (centered_warped.shape[0] - img_size[1]) // 2
    centered_warped[y_offset:y_offset+img_size[1], x_offset:x_offset+img_size[0]] = warped
    
    return centered_warped
prev_left_fit = []
prev_right_fit = []

# 定義滑動窗口法進行車道線檢測
def Slidingwin(binary_warped):
    global prev_left_fit
    global prev_right_fit

    # 創建一個輸出圖像來繪製和可視化結果
    OutImg = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # 計算二值圖像下半部分的直方圖
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    
    # 找到直方圖左半部分和右半部分的峰值
    midpoint = np.int32(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 設定滑動窗口的參數
    n_wins = 20
    win_height = binary_warped.shape[0] // n_wins
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    minpix = 50

    # 初始化當前位置，這些位置會在每個窗口中更新
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 創建空列表來接收左車道線和右車道線的像素索引
    left_lane_inds = []
    right_lane_inds = []

    # 一個接一個地步進窗口
    for win in range(n_wins):
        win_y_low = binary_warped.shape[0] - (win + 1) * win_height
        win_y_high = binary_warped.shape[0] - win * win_height
        win_xleft_low = leftx_current - margin
        win_xright_low = rightx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_high = rightx_current + margin

        # 在可視化圖像上繪製窗口
        cv2.rectangle(OutImg, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(OutImg, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # 識別窗口內的非零像素
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # 將這些索引添加到列表中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果找到的像素數超過minpix，則在它們的平均位置重新定位下一個窗口
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # 合併索引數組
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 提取左線和右線的像素位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 擬合每條線的二次多項式
    if leftx.size > 0 and lefty.size > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        prev_left_fit = left_fit
    else:
        left_fit = prev_left_fit

    if rightx.size > 0 and righty.size > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        prev_right_fit = right_fit
    else:
        right_fit = prev_right_fit

    return left_fit, right_fit

# 根據車道線擬合結果再次擬合
def fitFromLines(left_fit, right_fit, binary_warped):
    nonzero = binary_warped.nonzero()  # 獲取非零像素的位置
    nonzeroy = np.array(nonzero[0])  # 非零像素的y坐標
    nonzerox = np.array(nonzero[1])  # 非零像素的x坐標
    margin = 200  # 設定窗口的寬度
    
    # 計算多項式值
    left_fit_x = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2]
    right_fit_x = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2]

    # 找到左車道線像素
    left_lane_inds = ((nonzerox > (left_fit_x - margin)) & 
                      (nonzerox < (left_fit_x + margin)))
    
    # 找到右車道線像素
    right_lane_inds = ((nonzerox > (right_fit_x - margin)) & 
                       (nonzerox < (right_fit_x + margin)))
    
    # 獲取左車道線和右車道線的x, y坐標
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 只有當左或右車道線有足夠的點時，才擬合二次多項式
    if leftx.size > 0 and lefty.size > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if rightx.size > 0 and righty.size > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit

# 繪製車道線和區域
def DrawLines(img, img_w, left_fit, right_fit, perspective):
    # 創建黑色圖像和彩色圖像
    warp_zero = np.zeros_like(img_w).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # 獲取y坐標
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # 計算左車道線和右車道線x坐標
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 創建左車道線和右車道線點
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    # 合併車道線點並填充車道區域
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    # 透視變換車道區域並合併到原圖
    newwarp = warp(color_warp, perspective[1], perspective[0])
    result = cv2.addWeighted(img, 1, newwarp, 0.2, 0)

    # 創建黑色圖像，繪製左右車道線並透視變換
    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.polylines(color_warp_lines, np.int32([pts_right]), isClosed=False, color=(255, 255, 255), thickness=25)
    cv2.polylines(color_warp_lines, np.int32([pts_left]), isClosed=False, color=(0, 255, 255), thickness=25)
    newwarp_lines = warp(color_warp_lines, perspective[1], perspective[0])

    # 合併變換後的車道線到結果圖像
    result = cv2.addWeighted(result, 1, newwarp_lines, 0.8, 0)

    return result, newwarp_lines

cap = cv2.VideoCapture('testVideo.mp4') # 讀取視頻文件
fps = int(cap.get(cv2.CAP_PROP_FPS)) # 獲取視頻的幀率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 獲取視頻的寬度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 獲取視頻的高度

mov_avg_left = np.empty((0, 3)) # 初始化左車道線移動平均數組
mov_avg_right = np.empty((0, 3)) # 初始化右車道線移動平均數組

while True: # 開始視頻處理循環
    ret, img = cap.read() # 讀取一幀視頻
    if not ret: # 如果沒有讀取到幀
        print("Getting no frames") # 打印提示信息
        break # 跳出循環

    img = cv2.resize(img, (1280, 720)) # 調整圖像大小
    src = [(200, 720), (595, 450), (685, 450), (1100, 720)] # 設定源點
    dst = [(320, 720), (320, 0), (960, 0), (960, 720)] # 設定目標點
    H = np.float32([src]) # 將源點轉換為浮點數型別
    W = np.float32([dst]) # 將目標點轉換為浮點數型別
    result = ProcessImage(img) # 進行顏色處理
    rewslt = RegionOfInterest(result)
    img_w = warp(result, src, dst) # 進行透視變換
    cv2.imshow("warp", img_w)
    out_img = Slidingwin(img_w) # 使用滑動窗口方法
    left_fit, right_fit = out_img[0], out_img[1] # 獲取車道線擬合結果
    new_outimg = fitFromLines(left_fit, right_fit, img_w) # 再次擬合車道線
    left_fit, right_fit = new_outimg[0], new_outimg[1] # 更新車道線擬合結果

    if len(mov_avg_left) < MOV_AVG_LENGTH: # 如果移動平均數組長度小於設置值
        mov_avg_left = np.append(mov_avg_left, [left_fit], axis=0) # 添加左車道線擬合結果
        mov_avg_right = np.append(mov_avg_right, [right_fit], axis=0) # 添加右車道線擬合結果
    else:
        mov_avg_left = np.append(mov_avg_left[1:], [left_fit], axis=0) # 更新左車道線移動平均
        mov_avg_right = np.append(mov_avg_right[1:], [right_fit], axis=0) # 更新右車道線移動平均
        left_fit = [np.mean(mov_avg_left[:, 0]), np.mean(mov_avg_left[:, 1]), np.mean(mov_avg_left[:, 2])] # 計算左車道線移動平均
        right_fit = [np.mean(mov_avg_right[:, 0]), np.mean(mov_avg_right[:, 1]), np.mean(mov_avg_right[:, 2])] # 計算右車道線移動平均

    result, warp_img = DrawLines(img, img_w, left_fit, right_fit, (src, dst)) # 繪製車道線
    # curve_radius = curve(left_fit, right_fit, img.shape[0]) # 計算曲率半徑


    def slope(left, right, h):
        ym = 10 / 720 # y方向的尺度轉換
        xm = 4 / 1080 # x方向的尺度轉換
        left_slope = (2 * left[0] * h * ym + left[1]) / (ym * 2)  # 左車道線斜率
        right_slope = (2 * right[0] * h * ym + right[1]) / (ym * 2)  # 右車道線斜率
        return left_slope, right_slope # 返回左右車道線斜率
    left_slope, right_slope = slope(left_fit, right_fit, img.shape[0])

    try:
        if right_slope > 0:
            right_slope += 10
        print('left_slope ',left_slope)
        print('right_slope ',right_slope)
    except ZeroDivisionError:
        right_slope = float('inf') # 如果出現除零錯誤，設為無窮大

    font = cv2.FONT_HERSHEY_SIMPLEX # 設定字體
    if -10 < right_slope < 10:
        curve_direction = "Forward" 
    elif right_slope > 0:
        curve_direction = "Turn Left" 
    elif right_slope < 0:
        curve_direction = "Turn Right" 

    cv2.putText(result, curve_direction, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA) # 在圖像上顯示轉彎方向
    # out.write(result) # 寫入輸出視頻

    rows, cols = result.shape[:2] # 獲取圖像尺寸
    cv2.imshow('frame', result) # 顯示處理後的圖像

    if cv2.waitKey(1) & 0xFF == ord('q'): # 如果按下'q'鍵，退出循環
        break

cap.release() # 釋放視頻文件
# out.release() # 釋放輸出視頻文件
cv2.destroyAllWindows() # 關閉所有OpenCV窗口