import numpy as np # 導入NumPy庫
import cv2 # 導入OpenCV庫
import time # 導入時間庫
from scipy.stats import linregress

out_examples = 0 # 初始化變量out_examples為0
MOV_AVG_LENGTH = 5 # 設定移動平均的長度為5

# swheel = cv2.imread('steering_wheel_image.jpg') # 讀取方向盤圖片
# srows, scols, ch = swheel.shape # 獲取方向盤圖片的形狀
# smoothed_angle = 0 # 初始化平滑角度為0

# 定義顏色處理函數
def color(inpImage): 
    hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS) # 將輸入圖像轉換為HLS色彩空間
    lower_white = np.array([0, 160, 10]) # 設定白色的下界
    upper_white = np.array([255, 255, 255]) # 設定白色的上界
    mask = cv2.inRange(inpImage, lower_white, upper_white) # 對圖像進行閾值處理，提取白色部分
    hls_result = cv2.bitwise_and(inpImage, inpImage, mask=mask) # 使用掩膜進行位元運算
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY) # 將結果轉換為灰度圖像
    ret, thresh = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY) # 對灰度圖像進行閾值處理
    blur = cv2.GaussianBlur(thresh, (3, 3), 11) # 對二值圖像進行高斯模糊
    canny = cv2.Canny(blur, 40, 60) # 使用Canny邊緣檢測
    kernel = np.ones((9,9), dtype=np.uint8) 
    dilate = cv2.dilate(canny,kernel,iterations=1)
    cv2.imshow('re',dilate)
    return dilate # 返回邊緣檢測結果

# 定義Sobel邊緣檢測和二值化函數
def sobel_binary(img, sobel_kernel=7, mag_thresh=(3, 255), s_thresh=(170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) # 將圖像轉換為HLS色彩空間
    gray = hls[:, :, 1] # 提取亮度通道
    s_channel = hls[:, :, 2] # 提取飽和度通道
    sobel_binary = np.zeros(shape=gray.shape, dtype=bool) # 初始化Sobel二值圖像
    s_binary = sobel_binary # 初始化S通道二值圖像
    combined_binary = s_binary.astype(np.float32) # 初始化組合二值圖像
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # 計算Sobel X方向梯度
    sobely = 0 # Sobel Y方向梯度設為0
    sobel_abs = np.abs(sobelx**2 + sobely**2) # 計算Sobel梯度的絕對值
    sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs)) # 將梯度值轉換為8位圖像
    sobel_binary[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1 # 進行閾值處理
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1 # 進行閾值處理
    combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1 # 合併二值圖像
    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary)) # 將組合二值圖像轉換為8位圖像
    return combined_binary # 返回組合二值圖像

# 定義感興趣區域的遮罩函數
def region_of_interest(img):
    mask = np.zeros_like(img) # 創建與輸入圖像相同大小的黑色遮罩
    imshape = img.shape # 獲取圖像形狀
    vertices = np.array([[(0, imshape[0]), (390, 440), (880, 440), (imshape[1]-20, imshape[0])]], dtype=np.int32) # 定義多邊形頂點
    cv2.fillPoly(mask, vertices, 255) # 填充多邊形
    masked_image = cv2.bitwise_and(img, mask) # 應用遮罩
    cv2.imshow('a',mask)
    return masked_image # 返回遮罩後的圖像

# 定義透視變換函數
def warp(img, src, dst):
    src = np.float32([src]) # 將源點轉換為浮點數型別
    dst = np.float32([dst]) # 將目標點轉換為浮點數型別
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), dsize=img.shape[0:2][::-1], flags=cv2.INTER_LINEAR) # 進行透視變換

prev_left_fit = []
prev_right_fit = []
# 定義滑動窗口法進行車道線檢測
def sliding_windown(img_w):
    global prev_left_fit
    global prev_right_fit
    
    histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0) # 計算下半部分圖像的直方圖
    out_img = np.dstack((img_w, img_w, img_w)) * 255 # 創建彩色輸出圖像
    midpoint = np.int32(histogram.shape[0] / 2) # 計算直方圖中點
    leftx_base = np.argmax(histogram[:midpoint]) # 找到左車道線的基點
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # 找到右車道線的基點

    nwindows = 15 # 設定滑動窗口數量
    window_height = img_w.shape[0] // nwindows # 計算每個窗口的高度
    nonzero = img_w.nonzero() # 獲取非零像素的位置
    nonzeroy = np.array(nonzero[0]) # 非零像素的y坐標
    nonzerox = np.array(nonzero[1]) # 非零像素的x坐標
    leftx_current = leftx_base + 100 # 左車道線當前x坐標
    rightx_current = rightx_base # 右車道線當前x坐標
    margin = 100 # 設定窗口的寬度
    minpix = 50 # 設定最小像素數
    left_lane_inds = [] # 儲存左車道線像素索引
    right_lane_inds = [] # 儲存右車道線像素索引

    for window in range(nwindows):
        win_y_low = img_w.shape[0] - (window + 1) * window_height # 計算窗口的y坐標下界
        win_y_high = img_w.shape[0] - window * window_height # 計算窗口的y坐標上界
        win_xleft_low = leftx_current - margin # 計算左窗口的x坐標下界
        win_xleft_high = leftx_current + margin # 計算左窗口的x坐標上界
        win_xright_low = rightx_current - margin # 計算右窗口的x坐標下界
        win_xright_high = rightx_current + margin # 計算右窗口的x坐標上界
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) # 繪製左窗口
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) # 繪製右窗口
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0] # 找到左窗口內的像素
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0] # 找到右窗口內的像素
        left_lane_inds.append(good_left_inds) # 添加左車道線像素索引
        right_lane_inds.append(good_right_inds) # 添加右車道線像素索引
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds])) # 更新左車道線當前x坐標
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds])) # 更新右車道線當前x坐標
    cv2.imshow("slide", out_img)
    left_lane_inds = np.concatenate(left_lane_inds) # 合併左車道線像素索引
    right_lane_inds = np.concatenate(right_lane_inds) # 合併右車道線像素索引
    leftx = nonzerox[left_lane_inds] # 獲取左車道線x坐標
    lefty = nonzeroy[left_lane_inds] # 獲取左車道線y坐標
    rightx = nonzerox[right_lane_inds] # 獲取右車道線x坐標
    righty = nonzeroy[right_lane_inds] # 獲取右車道線y坐標
    left_fit = np.polyfit(lefty, leftx, 2) if leftx.size > 0 and lefty.size > 0 else prev_left_fit # 擬合左車道線二次多項式
    right_fit = np.polyfit(righty, rightx, 2) if rightx.size > 0 and righty.size > 0 else prev_right_fit # 擬合右車道線二次多項式

    if leftx.size > 0 and lefty.size > 0:
        prev_left_fit = left_fit
    
    if rightx.size > 0 and righty.size > 0:
        prev_right_fit = right_fit
    
    return left_fit, right_fit # 返回擬合結果

# 根據車道線擬合結果再次擬合
def fit_from_lines(left_fit, right_fit, img_w):
    nonzero = img_w.nonzero() # 獲取非零像素的位置
    nonzeroy = np.array(nonzero[0]) # 非零像素的y坐標
    nonzerox = np.array(nonzero[1]) # 非零像素的x坐標
    margin = 200 # 設定窗口的寬度
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin))) # 找到左車道線像素
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin))) # 找到右車道線像素
    leftx = nonzerox[left_lane_inds] # 獲取左車道線x坐標
    lefty = nonzeroy[left_lane_inds] # 獲取左車道線y坐標
    rightx = nonzerox[right_lane_inds] # 獲取右車道線x坐標
    righty = nonzeroy[right_lane_inds] # 獲取右車道線y坐標
    if leftx.any() == False or lefty.any() == False:
        return left_fit, right_fit
    else:
        left_fit = np.polyfit(lefty, leftx, 2) # 再次擬合左車道線二次多項式
        right_fit = np.polyfit(righty, rightx, 2) # 再次擬合右車道線二次多項式
    return left_fit, right_fit
     # 返回新的擬合結果

# 繪製車道線和區域
def draw_lines(img, img_w, left_fit, right_fit, perspective):
    warp_zero = np.zeros_like(img_w).astype(np.uint8) # 創建黑色圖像
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero)) # 創建彩色圖像

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0]) # 獲取y坐標
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2] # 計算左車道線x坐標
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2] # 計算右車道線x坐標

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]) # 創建左車道線點
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]); # 創建右車道線點
    pts = np.hstack((pts_left, pts_right)) # 合併車道線點

    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0)) # 填充車道區域

    newwarp = warp(color_warp, perspective[1], perspective[0]) # 透視變換車道區域
    result = cv2.addWeighted(img, 1, newwarp, 0.2, 0) # 合併原圖和車道區域

    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero)) # 創建黑色圖像
    cv2.polylines(color_warp_lines, np.int32([pts_right]), isClosed=False, color=(255, 255, 255), thickness=25) # 繪製右車道線
    cv2.polylines(color_warp_lines, np.int32([pts_left]), isClosed=False, color=(0, 255, 255), thickness=25) # 繪製左車道線
    newwarp_lines = warp(color_warp_lines, perspective[1], perspective[0]) # 透視變換車道線
    result = cv2.addWeighted(result, 1, newwarp_lines, 0.8, 0) # 合併原圖和車道線
   
    return result, newwarp_lines # 返回結果圖像和變換後的車道線圖像

# 計算曲率半徑
def curve(left, right, h):
    ym = 10 / 720 # y方向的尺度轉換
    xm = 4 / 1080 # x方向的尺度轉換
    left_fit_cr = [xm / (ym ** 2) * left[0], xm / ym * left[1], left[2]] # 左車道線擬合係數轉換
    right_fit_cr = [xm / (ym ** 2) * right[0], xm / ym * right[1], right[2]] # 右車道線擬合係數轉換
    curve_rad = ((1 + (2 * left_fit_cr[0] * h * ym + left_fit_cr[1]) ** 2) ** (3 / 2) / np.absolute(2 * left_fit_cr[0]) + (1 + (2 * right_fit_cr[0] * h * ym + right_fit_cr[1]) ** 2) ** (3 / 2) / np.absolute(2 * right_fit_cr[0])) # 計算曲率半徑
    return curve_rad # 返回曲率半徑

cap = cv2.VideoCapture('testVideo2.mp4') # 讀取視頻文件
fps = int(cap.get(cv2.CAP_PROP_FPS)) # 獲取視頻的幀率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 獲取視頻的寬度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 獲取視頻的高度
fourcc = cv2.VideoWriter_fourcc(*'XVID') # 設定視頻編碼格式
# out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height)) # 創建輸出視頻文件

mov_avg_left = np.empty((0, 3)) # 初始化左車道線移動平均數組
mov_avg_right = np.empty((0, 3)) # 初始化右車道線移動平均數組

while True: # 開始視頻處理循環
    ret, img = cap.read() # 讀取一幀視頻
    if not ret: # 如果沒有讀取到幀
        print("Getting no frames") # 打印提示信息
        break # 跳出循環

    src = [(200, 720), (595, 450), (685, 450), (1100, 720)] # 設定源點
    dst = [(320, 720), (320, 0), (960, 0), (960, 720)] # 設定目標點
    H = np.float32([src]) # 將源點轉換為浮點數型別
    W = np.float32([dst]) # 將目標點轉換為浮點數型別
    img = cv2.resize(img, (1280, 720)) # 調整圖像大小
    result = color(img) # 進行顏色處理
    rewslt = region_of_interest(result)
    img_w = warp(result, src, dst) # 進行透視變換
    cv2.imshow("warp", img_w)
    out_img = sliding_windown(img_w) # 使用滑動窗口方法
    left_fit, right_fit = out_img[0], out_img[1] # 獲取車道線擬合結果
    new_outimg = fit_from_lines(left_fit, right_fit, img_w) # 再次擬合車道線
    left_fit, right_fit = new_outimg[0], new_outimg[1] # 更新車道線擬合結果

    if len(mov_avg_left) < MOV_AVG_LENGTH: # 如果移動平均數組長度小於設置值
        mov_avg_left = np.append(mov_avg_left, [left_fit], axis=0) # 添加左車道線擬合結果
        mov_avg_right = np.append(mov_avg_right, [right_fit], axis=0) # 添加右車道線擬合結果
    else:
        mov_avg_left = np.append(mov_avg_left[1:], [left_fit], axis=0) # 更新左車道線移動平均
        mov_avg_right = np.append(mov_avg_right[1:], [right_fit], axis=0) # 更新右車道線移動平均
        left_fit = [np.mean(mov_avg_left[:, 0]), np.mean(mov_avg_left[:, 1]), np.mean(mov_avg_left[:, 2])] # 計算左車道線移動平均
        right_fit = [np.mean(mov_avg_right[:, 0]), np.mean(mov_avg_right[:, 1]), np.mean(mov_avg_right[:, 2])] # 計算右車道線移動平均

    result, warp_img = draw_lines(img, img_w, left_fit, right_fit, (src, dst)) # 繪製車道線
    curve_radius = curve(left_fit, right_fit, img.shape[0]) # 計算曲率半徑


    def slope(left, right, h):
        ym = 10 / 720 # y方向的尺度轉換
        xm = 4 / 1080 # x方向的尺度轉換
        left_slope = (2 * left[0] * h * ym + left[1]) / (ym * 2)  # 左車道線斜率
        right_slope = (2 * right[0] * h * ym + right[1]) / (ym * 2)  # 右車道線斜率
        return left_slope, right_slope # 返回左右車道線斜率
    left_slope, right_slope = slope(left_fit, right_fit, img.shape[0])

    try:
        radius = 5729.57795 / curve_radius # 計算轉彎半徑
        print('left_slope ',left_slope)
        print('right_slope ',right_slope)
    except ZeroDivisionError:
        right_slope = float('inf') # 如果出現除零錯誤，設為無窮大

    font = cv2.FONT_HERSHEY_SIMPLEX # 設定字體
    if radius > 0:
        curve_direction = "Curve Left" # 如果半徑大於0，顯示向左轉
    else:
        curve_direction = "Curve Right" # 如果半徑小於0，顯示向右轉

    cv2.putText(result, curve_direction, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA) # 在圖像上顯示轉彎方向
    # out.write(result) # 寫入輸出視頻

    rows, cols = result.shape[:2] # 獲取圖像尺寸
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), radius, 1) # 計算旋轉矩陣
    cv2.imshow('frame', result) # 顯示處理後的圖像

    if cv2.waitKey(1) & 0xFF == ord('q'): # 如果按下'q'鍵，退出循環
        break

cap.release() # 釋放視頻文件
# out.release() # 釋放輸出視頻文件
cv2.destroyAllWindows() # 關閉所有OpenCV窗口
