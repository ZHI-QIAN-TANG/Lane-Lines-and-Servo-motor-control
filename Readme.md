# 車道辨識與自動駕駛模擬
正式版檔案:  
Formaledition.py  
影片下載連結:
https://drive.google.com/drive/folders/1JEoK8s0YVq4H3Iv7BwtipS9eg0Hzr0OK?usp=sharing  
簡報:
https://www.canva.com/design/DAGHjwsMVwg/jnBIBUxfScyKUdu62H9dXg/view?utm_content=DAGHjwsMVwg&utm_campaign=designshare&utm_medium=link&utm_source=editor   
開發板: 樹梅派 Pi 4  
## 專案簡介
本專案旨在利用樹莓派硬體進行車道辨識與自動駕駛的模擬開發。透過圖像處理技術和自動駕駛演算法，本專案能夠辨識車道線並進行簡單的方向控制。 
![image](https://github.com/user-attachments/assets/7b91f954-cc70-4a71-a72e-9173d49d3092)

## 主要功能
- **影像讀取與處理**：讀取影像並進行格式轉換與預處理，確保影像的完整性。  
 ![image](https://github.com/user-attachments/assets/690954a5-5102-4ae4-aff3-a20d9b0b0455)
- **閥值處理**：將影像轉換成黑白二值化影像，提升車道線的辨識效果。  
  ![image](https://github.com/user-attachments/assets/9b5e43db-4525-42aa-9d7e-afd7ce37c7ca)
  ![image](https://github.com/user-attachments/assets/d0c5c651-6a5d-44ea-ae36-6f37e238fe6e)

- **高斯模糊**：對影像進行平滑處理，以減少雜訊干擾。  
  ![image](https://github.com/user-attachments/assets/b18263b0-7510-4ff3-82ee-0624ab1cfad9)

- **邊緣偵測**：利用邊緣檢測演算法找出圖像中的車道線。  
  ![image](https://github.com/user-attachments/assets/fcff66b0-2a51-4740-a0f2-a7d53517a910)
![image](https://github.com/user-attachments/assets/d3f22197-ad87-475a-9582-3ce03e4e0ba1)

- **投影轉換與 ROI 區域選取**：進行視角轉換，將影像變成俯視視角，並選取重點的感興趣區域 (ROI)。   
  ![image](https://github.com/user-attachments/assets/de8f99c7-cacb-48ec-ba58-5dfe84710e64)
![image](https://github.com/user-attachments/assets/cff122ce-9bb9-4bf5-8045-582de057d0af)

- **樣本點計算與曲線擬合**：找出車道線上的樣本點，並通過曲線擬合算法得到車道線的方程式。  
  ![image](https://github.com/user-attachments/assets/06b50e40-a063-4b01-9f4f-b987dcbfbadd)
![image](https://github.com/user-attachments/assets/0a165c0c-5912-4a3b-9ddd-696f47e998bd)

- **車道線繪製**：根據計算出的曲線斜率與角度，在影像中繪製車道線。  
  ![image](https://github.com/user-attachments/assets/c926d27f-7a73-4079-b89d-56e0453d5c8a)
- **自動駕駛控制**：利用曲線斜率和角度來判斷車輛行駛方向，實現左右轉彎控制。  
![image](https://github.com/user-attachments/assets/efee9ddb-2847-4f1d-9c04-d3bc7db64bfe)
![image](https://github.com/user-attachments/assets/4cb0e153-b9b6-4140-89c1-f69ab2fa7362)
## 專案成員與貢獻
- **411034018 王奕晨**：樹莓派硬體設計與邊緣檢測模組開發
- **411077018 唐知謙**：樹莓派環境搭建與軟硬體整合、車道線繪製
- **411077028 李泓逸**：ROI 區域選取、投影轉換與車道線檢測

## 主要技術與參考
- 本專案使用樹莓派 4 作為主要開發平台
- 參考書籍：**自駕車學習之路 (二)**，專注於車道線檢測與車道識別技術
### 開發流程
1. **讀取影像並進行預處理**：對影像進行格式轉換及去噪處理，提升影像品質。
2. **進行 ROI 區域選取與車道線檢測**：選取影像中的感興趣區域 (ROI)，並進行車道線檢測。
3. **利用曲線擬合與樣本點計算，辨識車道線**：通過計算影像中的樣本點來識別車道線，並進行曲線擬合。
4. **計算車道線的斜率與角度，繪製車道線**：根據曲線方程式計算車道線的斜率及角度，並在影像上進行繪製。
5. **根據車道線的資訊進行自動駕駛方向控制**：依據車道線角度資訊來判斷車輛方向，實現左右轉彎控制。

### 可能的改進方向
- **增強曲線擬合的準確性**：提升不同路況下的曲線擬合效果。
- **引入機器學習模型**：使用機器學習或深度學習模型來提升路徑規劃精度。
- **增加多種感測器 (如 LiDAR) 的支援**：提升系統對環境的感知能力。
  
## 安裝與使用指南
### 環境需求
- 樹莓派 4
- Python 3.7 或更高版本
- OpenCV
- Numpy

### 安裝步驟
1. 克隆此專案到本地環境：
   ```bash
   git clone https://github.com/your-repo/lane-recognition.git
### 執行方式
執行以下命令來啟動車道辨識模組並進行模擬：
```bash
python lane_detection.py 


