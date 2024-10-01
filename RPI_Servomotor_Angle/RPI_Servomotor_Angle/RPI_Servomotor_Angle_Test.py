import RPi.GPIO as GPIO
import time

# 設定伺服馬達的引腳
pin_servo = 5

# 設定GPIO模式
GPIO.setmode(GPIO.BCM)

# 設定伺服馬達的引腳為輸出模式
GPIO.setup(pin_servo, GPIO.OUT)
# 設定伺服馬達的轉動頻率
pwm_servo = GPIO.PWM(pin_servo, 50)  # 50Hz frequency
# 啟動PWM
pwm_servo.start(0)

# 關閉GPIO
def destroy():
    pwm_servo.stop()
    GPIO.cleanup()

# 將轉動角度換算成佔空比
def convert(x, i_m, i_M, o_m, o_M):
    return max(min(o_M, (x - i_m) * (o_M - o_m) // (i_M - i_m) + o_m), o_m)

# 設定伺服馬達的轉動角度
def set_direction(angle):
    # 使用 convert 函數將角度轉換成佔空比
    duty = convert(angle, 0, 180, 200, 1200) / 100.0  # 將範圍縮放到2-12之間
    pwm_servo.ChangeDutyCycle(duty)
    # 消除抖動
    time.sleep(0.5)
    pwm_servo.ChangeDutyCycle(0)
    print("角度=", angle, "-> duty=", duty)

if __name__ == "__main__":
    try:
        while True:
            for angle in [0, 90, 180, 90]:
                set_direction(angle)
                time.sleep(2)  # 等待2秒再轉到下一個角度
    except KeyboardInterrupt:
        pass
    finally:
        destroy()