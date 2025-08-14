import pygame

a =20
b =1

# เริ่มต้น Pygame และ mixer
pygame.init()
pygame.mixer.init()
Alarm = './Sound/error.mp3'
Tr = './Sound/Cap.mp3'
O_K = './Sound/OK.mp3'

def play_sound(status):
    """ฟังก์ชันสำหรับเล่นเสียง"""
    if status == "Error":
        sound = pygame.mixer.Sound(Alarm)  # ไฟล์เสียงสำหรับสถานะผิด
        # เล่นเสียงแบบวนลูป
        sound.play(loops=-1)
    elif status == "Cap":
        sound = pygame.mixer.Sound(Tr)  # ไฟล์เสียงสำหรับสถานะถูก
        # เล่นเสียงหนึ่งครั้ง
        sound.play()
    elif status == "OK":
        sound = pygame.mixer.Sound(O_K)  # ไฟล์เสียงสำหรับสถานะถูก
        # เล่นเสียงหนึ่งครั้ง
        sound.play()

def stop_sound():
    """ฟังก์ชันสำหรับหยุดเสียง"""
    pygame.mixer.stop()  # หยุดเสียงทั้งหมด

# while True:
#     if a+b > 1:
#         play_sound("True")
#     else:
#         play_sound("False")