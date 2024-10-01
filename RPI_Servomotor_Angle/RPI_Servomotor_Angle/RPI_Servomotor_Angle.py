import RPi.GPIO as GPIO
import time
import multiprocessing

# 設定伺服馬達的引腳
pin_servo = 5

# 設定GPIO模式
GPIO.setmode(GPIO.BCM)

# 設定伺服馬達的引腳為輸出模式
GPIO.setup(pin_servo, GPIO.OUT)

# 關閉GPIO
def destroy():
    GPIO.cleanup()

# 將轉動角度換算成佔空比
def convert(x, i_m, i_M, o_m, o_M):
    return max(min(o_M, (x - i_m) * (o_M - o_m) // (i_M - i_m) + o_m), o_m)

# 設定伺服馬達的轉動角度
def set_direction(angle):
    # 設定伺服馬達的轉動頻率
    pwm_servo = GPIO.PWM(pin_servo, 50)  # 50Hz frequency
    pwm_servo.start(0)
    
    # 使用 convert 函數將角度轉換成佔空比
    duty = convert(angle, 0, 180, 1000, 9000) / 1000.0  # 需要將1000-9000的範圍縮放到2-12之間
    pwm_servo.ChangeDutyCycle(duty)
    # 消除抖動
    time.sleep(0.3)
    pwm_servo.ChangeDutyCycle(0)
    print("角度=", angle, "-> duty=", duty)
    
    # 停止PWM訊號
    pwm_servo.stop()

if __name__ == "__main__":
    try:
        shared_angle = multiprocessing.Value('i', 0)  # 共享的角度值
        while True:
            # 從共享記憶體中讀取角度值
            angle = shared_angle.value
            # 根據讀取的角度值設定伺服馬達
            set_direction(angle)
            time.sleep(0.1)  # 等待一段時間再次讀取角度值

    finally:
        destroy()
