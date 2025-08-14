import tkinter as tk
import cv2
import threading
import time
from PIL import Image, ImageTk
import PIL
from ultralytics import YOLO
import cv2
import math
from PlaySound import *
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
pic_test = cv2.imread("Milk2_1_1_649.jpg")
model1(pic_test)

# ค่าตั้งต้น
DEFAULT_BG = "#f0f0f0"
DEFAULT_TEXT = "Input Waste"
DEFAULT_TEXT2 = "วางขยะได้เลย"

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
    # label2.configure(text=DEFAULT_TEXT2, font=("Arial", 28), bg=DEFAULT_BG)
    root.configure(bg=DEFAULT_BG)
    center_frame.configure(bg=DEFAULT_BG)
    label.configure(text=DEFAULT_TEXT, bg=DEFAULT_BG, fg="black")
    label2.configure(text=DEFAULT_TEXT2, bg=DEFAULT_BG, fg="black")
    label3.configure(text="", bg=DEFAULT_BG, fg="black")


def gpio_monitor_loop():
    while True:
        if GPIO.input(input_pin) == GPIO.LOW:
            print("🔵 Detected LOW on input_pin")
            time.sleep(1)
            handle_gpio_trigger()
            time.sleep(1)  # ป้องกันการ Trigger ซ้ำเร็วเกินไป


def handle_gpio_trigger():
    n = 0
    z, frame2 = check()
    if z == 4:
        n += 1
        if n == 1:
            play_sound('Error')
        root.configure(bg="red")
        center_frame.configure(bg="red")
        label.configure(text="ERROR", bg="red", fg="white")
        # label2.configure(text="", bg="red")
    if z == 3:
        n += 1
        if n == 1:
            play_sound('OK')
        root.configure(bg="green")
        center_frame.configure(bg="green")
        label.configure(text="ทิ้งตามสี", bg="green", fg="white")
        # label2.configure(text="", bg="green")
        GPIO.output(M3_up, GPIO.HIGH)
        time.sleep(0.4)
        GPIO.output(M3_up, GPIO.LOW)
    if z == 2:
        n += 1
        if n == 1:
            play_sound('OK')
        root.configure(bg="yellow")
        center_frame.configure(bg="yellow")
        label.configure(text="ทิ้งตามสี", bg="yellow", fg="Black")
        # label2.configure(text="", bg="yellow")
        GPIO.output(M2_up, GPIO.HIGH)
        time.sleep(0.4)
        GPIO.output(M2_up, GPIO.LOW)
    if z == 1:
        n += 1
        if n == 1:
            play_sound('OK')
        root.configure(bg="blue")
        center_frame.configure(bg="blue")
        label.configure(text="ทิ้งตามสี", bg="blue", fg="white")
        # label2.configure(text="", bg="blue")
        GPIO.output(M1_up, GPIO.HIGH)
        time.sleep(0.4)
        GPIO.output(M1_up, GPIO.LOW)
    if z == 0:
        n += 1
        if n == 1:
            play_sound('Cap')
        root.configure(bg="red")
        center_frame.configure(bg="red")
        label.configure(text="เอาฝาออกก่อนครับ", bg="red", fg="white")
    root.after(2000, lambda: start_countdown(5, z))

    time.sleep(2)


def handle_keypress(event):
    global label
    if event.char.lower() == 's':
        n = 0
        z, frame2 = check()
        if z == 4:
            n += 1
            if n == 1:
                play_sound('Error')
            root.configure(bg="red")
            center_frame.configure(bg="red")
            label.configure( text="ERROR", bg="red", fg="white")
            # label2.configure(text="", bg="red")
        if z == 3:
            n += 1
            if n == 1:
                play_sound('OK')
            root.configure(bg="green")
            center_frame.configure(bg="green")
            label.configure(text="ทิ้งตามสี", bg="green", fg="white")
            # label2.configure(text="", bg="green")
            GPIO.output(M3_up, GPIO.HIGH)
            time.sleep(0.4)
            GPIO.output(M3_up, GPIO.LOW)
        if z == 2:
            n += 1
            if n == 1:
                play_sound('OK')
            root.configure(bg="yellow")
            center_frame.configure(bg="yellow")
            label.configure(text="ทิ้งตามสี", bg="yellow", fg="Black")
            # label2.configure(text="", bg="yellow")
            GPIO.output(M2_up, GPIO.HIGH)
            time.sleep(0.4)
            GPIO.output(M2_up, GPIO.LOW)
        if z == 1:
            n += 1
            if n == 1:
                play_sound('OK')
            root.configure(bg="blue")
            center_frame.configure(bg="blue")
            label.configure(text="ทิ้งตามสี", bg="blue", fg="white")
            # label2.configure(text="", bg="blue")
            GPIO.output(M1_up, GPIO.HIGH)
            time.sleep(0.4)
            GPIO.output(M1_up, GPIO.LOW)
        if z == 0:
            n += 1
            if n == 1:
                play_sound('Cap')
            root.configure(bg="red")
            center_frame.configure(bg="red")
            label.configure( text="เอาฝาออกก่อนครับ", bg="red", fg="white")
        root.after(2000, lambda: start_countdown(5, z))

def reset_gui():
    label.config(image='', text=DEFAULT_TEXT, bg=DEFAULT_BG)


def update_image():
    global label
    global label
    if frame is not None:
        img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        label.config(image=img)
        label.image = img
        label.place(relx=0.5, rely=0.15, anchor="center")  # แสดงภาพ
    else:
        label.config(image='')  # เคลียร์ภาพ
        label.place_forget()  # ซ่อน camera_label

    label.config(image='', text='')  # เคลียร์ภาพ
    label.place_forget()  # ซ่อน camera_label


def start_countdown(seconds, z):
    if z == 0:
        status = "Bottle Cap"
        status2 = "ฝาขวดน้ำ"
        col = "red"
        col2 = "white"

    if z == 1:
        status = "Plastic Waste"
        status2 = "ขวดพลาสติก"
        col = "blue"
        col2 = "white"

    if z == 2:
        status = "General Waste"
        status2 = "ขยะทั่วไป"
        col = "yellow"
        col2 = "black"
    if z == 3:
        status = "Glass/Metal Waste"
        status2 = "ขวดแก้ว/โลหะ"
        col = "green"
        col2 = "white"

    if z == 4:
        status = "ERROR"
        status2 = ""
        col = "red"
        col2 = "white"

    if seconds > 0:
        label.place(relx=0.5, rely=0.25, anchor="center")
        label.configure(text=f"{status}", bg=col,fg=col2,font=("Arial", 48))
        label2.configure(text=f"{status2}", bg=col, fg=col2,font=("Arial", 48))
        label3.configure(text=f"กลับสู่หน้าจอหลักใน {seconds} วินาที", bg=col, fg=col2)
        root.after(1000, start_countdown, seconds - 1, z)
    else:
        reset_to_default(z)


def check():
    global frame, label,imgtk_ref
    a = 0
    z = 0
    b = 3
    err = 0
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
            cv2.putText(frame, f'{class_name} , {confidence}', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 255, 255), 2)
            if class_name == 'Cap':
                b = 0
            if class_name == 'Not_cap':
                b = 1
            if class_name == 'Mansome' or class_name == 'Honey' or class_name == 'Crystal':
                print(class_name)
                a = 1
            if class_name == 'M100' or class_name == 'Vitamilk':
                print(class_name)
                a = 2
            if class_name == "Milk2" or class_name == 'Milk1':
                z = 2
                err = 1
            if class_name == "Coke":
                z = 3
                err = 1
    if a == 1:
        if b == 1:
            print("No cap")
            z = 1
    if a == 2:
        if b == 1:
            print("No cap")
            z = 3
    if b == 3 and err != 1:
        z = 4

    cv2.imwrite("test.jpg", frame)
    print(f"b = {b}  err = {err}")
    print(f"z = {z}")
    frame = cv2.resize(frame, (800, 450))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.config(image=imgtk)
    label.imgtk = imgtk
    label.place(relx=0.5, rely=0.45, anchor="center")
    label2.configure(text=f"")
    root.after(2000, reset_gui)
    return z, frame


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
    print("❌ เริ่ม")
    camera_running = True
    while camera_running:
        # frame = pic_test
        # time.sleep(0.5)
        ret, frame = cap.read()
        # print("====================================")
        if not ret:
            print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
            break
        # คุณสามารถเพิ่มการประมวลผล frame ได้ที่นี่ (ถ้าต้องการ)
        time.sleep(0.03)  # ประมาณ 30 FPS

# UI Elements
center_frame = tk.Frame(root, width=800, height=400, bg=DEFAULT_BG)
center_frame.pack(expand=True)

label = tk.Label(center_frame, text=DEFAULT_TEXT, font=("Arial", 36), bg=DEFAULT_BG)
label.place(relx=0.5, rely=0.25, anchor="center")
label2 = tk.Label(center_frame, text=DEFAULT_TEXT2, font=("Arial", 36), bg=DEFAULT_BG)
label2.place(relx=0.5, rely=0.45, anchor="center")
label3 = tk.Label(center_frame, text="", font=("Arial", 28), bg=DEFAULT_BG)
label3.place(relx=0.5, rely=0.85, anchor="center")

exit_button = tk.Button(root, text="ออกโปรแกรม", font=("Arial", 18), command=quit_app)
exit_button.pack(side="bottom", pady=0)


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