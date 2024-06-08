import cv2
import numpy as np

def nothing(x):
    pass

# 創建窗口
cv2.namedWindow('Edges')

# 創建滑動條來調整低閾值和高閾值
cv2.createTrackbar('Low Threshold', 'Edges', 0, 255, nothing)
cv2.createTrackbar('High Threshold', 'Edges', 0, 255, nothing)
cv2.createTrackbar('Kernel Size', 'Edges', 1, 10, nothing)
cv2.createTrackbar('Blur Amount', 'Edges', 0, 20, nothing)
cv2.createTrackbar('Playback Speed', 'Edges', 1, 10, nothing)

# 設置初始值
cv2.setTrackbarPos('Low Threshold', 'Edges', 100)
cv2.setTrackbarPos('High Threshold', 'Edges', 200)
cv2.setTrackbarPos('Kernel Size', 'Edges', 1)
cv2.setTrackbarPos('Blur Amount', 'Edges', 1)
cv2.setTrackbarPos('Playback Speed', 'Edges', 1)

# 讀取影片
video_path = 'testVideo.mp4'  # 影片路徑
cap = cv2.VideoCapture(video_path)

# 檢查是否成功打開影片
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 獲取影片的幀率
fps = int(cap.get(cv2.CAP_PROP_FPS))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換為灰度圖像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 獲取滑動條的當前位置
    low_threshold = cv2.getTrackbarPos('Low Threshold', 'Edges')
    high_threshold = cv2.getTrackbarPos('High Threshold', 'Edges')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Edges') * 2 + 1  # 保證是奇數
    blur_amount = cv2.getTrackbarPos('Blur Amount', 'Edges')
    playback_speed = cv2.getTrackbarPos('Playback Speed', 'Edges')

    # 應用高斯模糊
    blurred_frame = cv2.GaussianBlur(gray_frame, (blur_amount*2+1, blur_amount*2+1), 0)

    # 應用Canny邊緣檢測
    edges = cv2.Canny(blurred_frame, low_threshold, high_threshold)

    # 形態學操作來加強邊緣
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 顯示結果
    cv2.imshow('Edges', edges)
    
    # 計算延遲時間
    delay = int(1000 / (fps * playback_speed))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()