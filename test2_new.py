import tkinter as tk
import cv2
import threading
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import math
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

input_pin = 12
GPIO.setup(input_pin, GPIO.IN)
# GPIO.input(input_pin, GPIO.HIGH)
M1_up = 4
GPIO.setup(M1_up, GPIO.OUT)
GPIO.output(M1_up, GPIO.LOW)
M1_down = 17
GPIO.setup(M1_down, GPIO.OUT)
GPIO.output(M1_down, GPIO.LOW)
M2_up = 27
GPIO.setup(M2_up, GPIO.OUT)
GPIO.output(M2_up, GPIO.LOW)
M2_down = 22
GPIO.setup(M2_down, GPIO.OUT)
GPIO.output(M2_down, GPIO.LOW)
M3_up = 5
GPIO.setup(M3_up, GPIO.OUT)
GPIO.output(M3_up, GPIO.LOW)
M3_down = 6
GPIO.setup(M3_down, GPIO.OUT)
GPIO.output(M3_down, GPIO.LOW)

model1 = YOLO("./model/All.pt")
pic_test = cv2.imread("Vitamilk_1_1_302.jpg")
model1(pic_test)

# ค่าตั้งต้น
DEFAULT_BG = "#f0f0f0"
DEFAULT_TEXT = "วางขวดในช่องที่ระบุ"

# สร้างหน้าต่างหลัก
root = tk.Tk()
root.title("Bottle Placement")
root.geometry("800x450")
root.resizable(False, False)
root.configure(bg=DEFAULT_BG)


def quit_app():
    # ปิดกล้องก่อนออก
    global camera_running
    GPIO.cleanup()
    camera_running = False
    if cap.isOpened():
        cap.release()
    root.destroy()


def reset_to_default(z):
    if z == 1:
        GPIO.output(M1_down, GPIO.HIGH)
        time.sleep(0.45)
        GPIO.output(M1_down, GPIO.LOW)
    if z == 2:
        GPIO.output(M2_down, GPIO.HIGH)
        time.sleep(0.45)
        GPIO.output(M2_down, GPIO.LOW)
    if z == 3:
        GPIO.output(M3_down, GPIO.HIGH)
        time.sleep(0.45)
        GPIO.output(M3_down, GPIO.LOW)
    root.configure(bg=DEFAULT_BG)
    center_frame.configure(bg=DEFAULT_BG)
    label.configure(text=DEFAULT_TEXT, bg=DEFAULT_BG, fg="black")


def gpio_monitor_loop():
    while True:
        if GPIO.input(input_pin) == GPIO.LOW:
            print("🔵 Detected LOW on input_pin")
            time.sleep(1)
            handle_gpio_trigger()
            time.sleep(1)  # ป้องกันการ Trigger ซ้ำเร็วเกินไป


def handle_gpio_trigger():
    z = check()
    if z == 3:

        root.configure(bg="green")
        center_frame.configure(bg="green")
        label.configure(text="ทิ้งตามสี", bg="green", fg="white")
        GPIO.output(M3_up, GPIO.HIGH)
        time.sleep(0.45)
        GPIO.output(M3_up, GPIO.LOW)
    elif z == 2:

        root.configure(bg="yellow")
        center_frame.configure(bg="yellow")
        label.configure(text="ทิ้งตามสี", bg="yellow", fg="Black")
        GPIO.output(M2_up, GPIO.HIGH)
        time.sleep(0.45)
        GPIO.output(M2_up, GPIO.LOW)
    elif z == 1:

        root.configure(bg="blue")
        center_frame.configure(bg="blue")
        label.configure(text="ทิ้งตามสี", bg="blue", fg="white")
        GPIO.output(M1_up, GPIO.HIGH)
        time.sleep(0.45)
        GPIO.output(M1_up, GPIO.LOW)
    elif z == 0:
        root.configure(bg="red")
        center_frame.configure(bg="red")
        label.configure(text="เอาฝาออกก่อนครับ", bg="red", fg="white")
    root.after(6000, lambda: reset_to_default(z))
    start_countdown(5, z)

    time.sleep(2)


def handle_keypress(event):
    if event.char.lower() == 's':
        z = check()
        if z == 3:
            root.configure(bg="green")
            center_frame.configure(bg="green")
            label.configure(text="ทิ้งตามสี", bg="green", fg="white")
            GPIO.output(M3_up, GPIO.HIGH)
            time.sleep(0.4)
            GPIO.output(M3_up, GPIO.LOW)
        if z == 2:
            root.configure(bg="yellow")
            center_frame.configure(bg="yellow")
            label.configure(text="ทิ้งตามสี", bg="yellow", fg="Black")
            GPIO.output(M2_up, GPIO.HIGH)
            time.sleep(0.4)
            GPIO.output(M2_up, GPIO.LOW)
        if z == 1:
            root.configure(bg="blue")
            center_frame.configure(bg="blue")
            label.configure(text="ทิ้งตามสี", bg="blue", fg="white")
            GPIO.output(M1_up, GPIO.HIGH)
            time.sleep(0.4)
            GPIO.output(M1_up, GPIO.LOW)
        if z == 0:
            root.configure(bg="red")
            center_frame.configure(bg="red")
            label.configure(text="เอาฝาออกก่อนครับ", bg="red", fg="white")
        # root.after(5000, lambda: reset_to_default(z))
        start_countdown(5, z)


def start_countdown(seconds, z):
    if seconds > 0:
        label.configure(text=f"กลับสู่หน้าจอหลักใน {seconds} วินาที")
        root.after(1000, start_countdown, seconds - 1, z)
    else:
        reset_to_default(z)


def check():
    global frame
    a = 0
    z = 0
    b = 0
    results = model1(frame, conf=0.7)
    for i in results:
        classes_names1 = i.names
        boxes = i.boxes
        # print(boxes)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x, y, w, h = box.xywh[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classes_names1[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, f'{class_name} , {confidence}', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)
            if class_name == 'Cap':
                b = 0
            if class_name == 'Not_Cap':
                b = 1
            if class_name == 'Mansome' or class_name == 'Honey' or class_name == 'Crystal':
                print(class_name)
                a = 1
            if class_name == 'M100' or class_name == 'Vitamilk':
                print(class_name)
                a = 2
            if class_name == "Milk2" or class_name == 'Milk1':
                z = 2
            if class_name == "Coke":
                z = 3
    if a == 1:
        if b == 1:
            print("No cap")
            z = 1
    if a == 2:
        if b == 1:
            print("No cap")
            z = 3

    cv2.imwrite("test.jpg", frame)
    print(f"z = {z}")
    return z


# ---- กล้อง ----
def camera_loop():
    global cap, camera_running, frame
    width = 1920
    height = 1080
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # ตั้งค่าความกว้างของเฟรมวิดีโอ
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #    ตั้งค่าความสูงของเฟรมวิดีโอ
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("❌ ไม่พบกล้อง")
        return

    camera_running = True
    while camera_running:
        # frame = pic_test
        # time.sleep(0.5)
        ret, frame = cap.read()
        print("====================================")
        if not ret:
            print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
            break
        # คุณสามารถเพิ่มการประมวลผล frame ได้ที่นี่ (ถ้าต้องการ)
        time.sleep(0.03)  # ประมาณ 30 FPS

# UI Elements
center_frame = tk.Frame(root, width=800, height=400, bg=DEFAULT_BG)
center_frame.pack(expand=True)

label = tk.Label(center_frame, text=DEFAULT_TEXT, font=("Arial", 28), bg=DEFAULT_BG)
label.place(relx=0.5, rely=0.5, anchor="center")

exit_button = tk.Button(root, text="ออกโปรแกรม", font=("Arial", 18), command=quit_app)
exit_button.pack(side="bottom", pady=20)


camera_thread = threading.Thread(target=camera_loop, daemon=True)
camera_thread.start()
# ผูก event
root.bind("<Key>", handle_keypress)





# เริ่มกล้องใน thread แยก

gpio_thread = threading.Thread(target=gpio_monitor_loop, daemon=True)
gpio_thread.start()

# เริ่ม GUI
root.mainloop()

import signal
import sys


def signal_handler(sig, frame):
    print("⚠️ กำลังปิดระบบและ cleanup GPIO...")
    GPIO.cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)