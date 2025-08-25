# app.py
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import math
import glob
import shutil
import threading
import queue
import csv
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Optional (used when training/testing)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Optional (for graphs)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Helper utilities ----------
APP_ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECTS_DIR = os.path.join(APP_ROOT, "projects")
EXPORTS_DIR = os.path.join(APP_ROOT, "exports")
RUNS_DIR = os.path.join(APP_ROOT, "runs")
LOGS_DIR = os.path.join(APP_ROOT, "logs")

os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def safe_filename(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in (" ", "_", "-", ".")).rstrip()

def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def pil_to_ctk_image(pil_img, size=None):
    if size:
        pil_img = pil_img.copy()
        pil_img.thumbnail(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

def pil_from_cv2(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_img)

def cv2_from_pil(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def compute_yolo_bbox(txt_w, txt_h, x1, y1, x2, y2):
    # normalize bounding box [x_center, y_center, w, h]
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    cx = (x1 + x2) / 2.0 / txt_w
    cy = (y1 + y2) / 2.0 / txt_h
    w = abs(x2 - x1) / txt_w
    h = abs(y2 - y1) / txt_h
    return cx, cy, w, h

def rescale_points(pts, old_w, old_h, new_w, new_h):
    sx = new_w / max(1, old_w)
    sy = new_h / max(1, old_h)
    return [(int(x * sx), int(y * sy)) for x, y in pts]

def mask_to_polygons(mask):
    # mask: HxW, values 0 or 255
    # returns list of polygons (each polygon is list of (x,y))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) >= 3:
            poly = [(int(p[0][0]), int(p[0][1])) for p in cnt]
            polys.append(poly)
    return polys

def normalize_polygon(poly, w, h):
    # YOLO segment expects x1 y1 x2 y2 ... normalized (0-1)
    flat = []
    for x, y in poly:
        flat.append(str(round(x / w, 6)))
        flat.append(str(round(y / h, 6)))
    return flat

def draw_boxes_on_pil(pil_img, boxes, color=(0,255,0), width=2):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for (cls_id, x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=width)
        draw.text((x1+2, y1+2), f"{cls_id}", fill=(255,255,0))
    return img

def overlay_mask_on_pil(pil_img, mask, color=(255, 0, 0), alpha=0.3):
    if mask is None:
        return pil_img
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    overlay_np = np.array(overlay)
    mask_np = np.array(mask)
    r,g,b = color
    overlay_np[mask_np > 0] = [r, g, b, int(255*alpha)]
    overlay = Image.fromarray(overlay_np, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


# ---------- Global app data to support session logging ----------
class SessionLogger:
    def __init__(self):
        self.projects_created = []
        self.train_runs = []

    def add_project(self, name, count):
        self.projects_created.append({"project_name": name, "images_count": count, "timestamp": datetime.now().isoformat()})

    def add_train(self, mode, model_size, num_classes, class_names, epochs, batch, imgsz, project_name, start_time, end_time):
        self.train_runs.append({
            "mode": mode,
            "model_size": model_size,
            "num_classes": num_classes,
            "class_names": class_names,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "project_name": project_name,
            "duration_sec": (end_time - start_time),
            "start": datetime.fromtimestamp(start_time).isoformat(),
            "end": datetime.fromtimestamp(end_time).isoformat(),
        })

    def write_on_exit(self):
        data = {
            "session_start": getattr(self, "_session_start", None) or datetime.now().isoformat(),
            "session_end": datetime.now().isoformat(),
            "projects_count": len(self.projects_created),
            "projects": self.projects_created,
            "train_runs": self.train_runs,
        }
        path = os.path.join(LOGS_DIR, f"session_{now_str()}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path


RESOLUTIONS = {
    "4K (3840x2160)": (3840, 2160),
    "2K (2560x1440)": (2560, 1440),
    "Full HD (1920x1080)": (1920, 1080),
    "HD (1280x720)": (1280, 720),
    "1024x768": (1024, 768),
    "800x600": (800, 600),
    "VGA (640x480)": (640, 480),
    "320x240": (320, 240)
}

def get_max_resolution(cap):
    for width, height in RESOLUTIONS.values():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == width and actual_h == height:
            return width, height
    return 640, 480


class CaptureTab(ctk.CTkFrame):
    def __init__(self, master, session_logger, on_project_saved, camera_index=0):
        super().__init__(master)
        self.session_logger = session_logger
        self.on_project_saved = on_project_saved
        self.camera_index = camera_index

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # Left: webcam feed
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Right: controls + list
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # --- ปุ่มควบคุม ---
        btn_frame = ctk.CTkFrame(right)
        btn_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=(4,0))
        btn_frame.grid_columnconfigure((0,1,2), weight=1)

        self.shot_btn = ctk.CTkButton(btn_frame, text="ถ่ายภาพ", command=self.capture_frame)
        self.shot_btn.grid(row=0, column=0, padx=4, pady=6, sticky="ew")

        self.reset_btn = ctk.CTkButton(btn_frame, text="รีเซ็ต", fg_color="#A33", hover_color="#922", command=self.reset_list)
        self.reset_btn.grid(row=0, column=1, padx=4, pady=6, sticky="ew")

        self.save_btn = ctk.CTkButton(btn_frame, text="บันทึก", fg_color="#2b7", hover_color="#279", command=self.save_project)
        self.save_btn.grid(row=0, column=2, padx=4, pady=6, sticky="ew")

        # --- Drop-down เลือกความละเอียด ---
        self.resolution_var = ctk.StringVar(value="เลือกความละเอียด")
        self.resolution_menu = ctk.CTkOptionMenu(
            right,
            variable=self.resolution_var,
            values=list(RESOLUTIONS.keys()),
            command=self.change_resolution
        )
        self.resolution_menu.grid(row=1, column=0, padx=6, pady=(6,0), sticky="ew")

        self.count_label = ctk.CTkLabel(right, text="รูปทั้งหมด: 0")
        self.count_label.grid(row=2, column=0, sticky="w", padx=6, pady=(6,0))

        self.scroll = ctk.CTkScrollableFrame(right, label_text="รายการรูปที่ถ่าย")
        self.scroll.grid(row=3, column=0, sticky="nsew", padx=4, pady=6)
        self.thumb_widgets = []

        # Internal state
        self.captured = []
        self.cap = None
        self.running = True

        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._video_thread.start()

    def _video_loop(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถเปิดกล้องได้")
                return

            max_w, max_h = get_max_resolution(self.cap)
            print(f"ใช้ความละเอียดสูงสุด: {max_w}x{max_h}")

            last_frame = None
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    last_frame = frame
                if last_frame is not None:
                    pil = pil_from_cv2(last_frame)
                    img = pil_to_ctk_image(pil, size=(1280, 720))
                    self.video_label.configure(image=img)
                    self.video_label.image = img
                time.sleep(0.03)
        finally:
            if self.cap:
                self.cap.release()

    def change_resolution(self, choice):
        """เปลี่ยนความละเอียดโดยปิดและเปิดกล้องใหม่"""
        if self.cap:
            self.cap.release()
            self.cap = None

        width, height = RESOLUTIONS[choice]

        # เปิดกล้องใหม่
        new_cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # ใช้ CAP_DSHOW บน Windows จะควบคุมได้ดีกว่า MSMF
        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not new_cap.isOpened():
            messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถเปิดกล้องได้")
            return

        if (actual_w, actual_h) != (width, height):
            messagebox.showwarning(
                "ไม่รองรับ",
                f"กล้องไม่รองรับ {width}x{height} (ใช้ {actual_w}x{actual_h} แทน)"
            )

        self.cap = new_cap

    def capture_frame(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("กล้องไม่พร้อม", "ไม่สามารถเข้าถึงกล้องได้")
            return
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("ข้อผิดพลาด", "ถ่ายภาพไม่สำเร็จ")
            return
        pil = pil_from_cv2(frame)
        self.captured.append(pil)
        self._add_thumbnail(pil)
        self.count_label.configure(text=f"รูปทั้งหมด: {len(self.captured)}")

    def stop(self):
        self.running = False
        if self._video_thread.is_alive():
            self._video_thread.join()

    def _add_thumbnail(self, pil_img):
        idx = len(self.thumb_widgets)
        thumb = pil_to_ctk_image(pil_img, size=(220, 140))
        frame = ctk.CTkFrame(self.scroll)
        frame.grid_columnconfigure(1, weight=1)
        frame.pack(fill="x", padx=6, pady=6)

        lbl = ctk.CTkLabel(frame, image=thumb, text="")
        lbl.image = thumb
        lbl.grid(row=0, column=0, rowspan=2, padx=6, pady=6, sticky="w")

        name = ctk.CTkLabel(frame, text=f"ภาพที่ {idx+1}")
        name.grid(row=0, column=1, sticky="w", padx=4)
        size = ctk.CTkLabel(frame, text=f"{pil_img.width}x{pil_img.height}")
        size.grid(row=1, column=1, sticky="w", padx=4)

        self.thumb_widgets.append(frame)

    def reset_list(self):
        if not self.captured:
            return
        if not messagebox.askyesno("ยืนยัน", "ต้องการลบภาพทั้งหมดใช่หรือไม่?"):
            return
        self.captured.clear()
        for w in self.thumb_widgets:
            w.destroy()
        self.thumb_widgets.clear()
        self.count_label.configure(text="รูปทั้งหมด: 0")

    def save_project(self):
        if not self.captured:
            messagebox.showwarning("ไม่มีรูป", "ยังไม่มีรูปสำหรับบันทึก")
            return
        dialog = ctk.CTkInputDialog(text="ตั้งชื่อโปรเจกต์", title="บันทึกรูปภาพ")
        name = dialog.get_input()
        if not name:
            return
        name = safe_filename(name)
        proj_dir = os.path.join(PROJECTS_DIR, name)
        if os.path.exists(proj_dir):
            if not messagebox.askyesno("ซ้ำชื่อ", "มีโปรเจกต์ชื่อนี้อยู่แล้ว ต้องการเขียนทับหรือไม่?"):
                return
            shutil.rmtree(proj_dir)
        os.makedirs(proj_dir, exist_ok=True)
        img_dir = os.path.join(proj_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        # Save images
        for i, im in enumerate(self.captured, start=1):
            im.save(os.path.join(img_dir, f"img_{i:04d}.jpg"), quality=95)
        self.session_logger.add_project(name, len(self.captured))
        self.on_project_saved(name, proj_dir)
        messagebox.showinfo("สำเร็จ", f"บันทึก {len(self.captured)} รูป ไปยังโปรเจกต์: {name}")
        # Keep images in memory for further work; don't reset automatically

    def destroy(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        super().destroy()


# ---------- Tab 2: Labeling ----------
class LabelTab(ctk.CTkFrame):
    def __init__(self, master, get_projects_callable):
        super().__init__(master)
        self.get_projects = get_projects_callable

        # Layout: Left (selectors + thumbnails), Center (viewer), Right (tools)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=6)
        self.grid_columnconfigure(2, weight=3)

        # Left panel
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)

        sel_frame = ctk.CTkFrame(left)
        sel_frame.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        sel_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(sel_frame, text="เลือกโปรเจกต์: ").grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.project_combo = ctk.CTkComboBox(sel_frame, values=self._list_project_names(), command=self._on_project_selected)
        self.project_combo.grid(row=0, column=1, padx=4, pady=6, sticky="ew")

        self.browse_btn = ctk.CTkButton(sel_frame, text="เลือกโฟลเดอร์ภายนอก", command=self._browse_external)
        self.browse_btn.grid(row=0, column=2, padx=4, pady=6, sticky="e")

        self.thumb_scroll = ctk.CTkScrollableFrame(left, label_text="รูปทั้งหมดในโปรเจกต์")
        self.thumb_scroll.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)

        # Center viewer (scrollable with zoom)
        center = ctk.CTkFrame(self)
        center.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)

        # Use CTkCanvas via tkinter Canvas embedded (CTk doesn't have native Canvas)
        import tkinter as tk
        self.tkcanvas = tk.Canvas(center, bg="#222222", highlightthickness=0, scrollregion=(0,0,2000,2000))
        self.tkcanvas.grid(row=0, column=0, sticky="nsew")

        # Add scrollbars
        vbar = ctk.CTkScrollbar(center, command=self.tkcanvas.yview)
        vbar.grid(row=0, column=2, sticky="ns")
        hbar = ctk.CTkScrollbar(center, orientation="horizontal", command=self.tkcanvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        self.tkcanvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        # Right tools
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)
        right.grid_columnconfigure(0, weight=1)

        self.tool_label = ctk.CTkLabel(right, text="เครื่องมือทำ Label")
        self.tool_label.grid(row=0, column=0, sticky="w", padx=6, pady=(6,0))

        self.tool_var = ctk.StringVar(value="box")
        self.tool_seg = ctk.CTkRadioButton(right, text="Brush Marker (Segment)", variable=self.tool_var, value="segment")
        self.tool_box = ctk.CTkRadioButton(right, text="Bounding Box (Detect)", variable=self.tool_var, value="box")
        self.tool_seg.grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.tool_box.grid(row=2, column=0, sticky="w", padx=6, pady=4)

        size_frame = ctk.CTkFrame(right)
        size_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=6)
        ctk.CTkLabel(size_frame, text="ขนาดหัวแปรง:").grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.brush_size = ctk.CTkSlider(size_frame, from_=3, to=80, number_of_steps=77)
        self.brush_size.set(18)
        self.brush_size.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        size_frame.grid_columnconfigure(1, weight=1)

        class_frame = ctk.CTkFrame(right)
        class_frame.grid(row=4, column=0, sticky="ew", padx=6, pady=6)
        ctk.CTkLabel(class_frame, text="หมายเลข Class:").grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.class_entry = ctk.CTkEntry(class_frame, placeholder_text="เช่น 0, 1, 2")
        self.class_entry.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        class_frame.grid_columnconfigure(1, weight=1)
        # — class palette UI —
        self.active_class = None
        self.class_buttons = {}  # cls_id -> button

        self.class_palette = ctk.CTkFrame(right)
        self.class_palette.grid(row=5, column=0, sticky="ew", padx=6, pady=(0, 6))
        self.class_palette.grid_columnconfigure(0, weight=1)

        # Enter เพื่อเพิ่มคลาส
        def _bind_add_class(event):
            self._add_class_from_entry()

        self.class_entry.bind("<Return>", _bind_add_class)

        # คลิกที่อื่นให้โฟกัสออกจากช่อง เพื่อไม่ให้พิมพ์ลงช่องโดยไม่ได้ตั้งใจ
        # self.bind_all("<Button-1>", lambda e: (self.focus_set() if e.widget != self.class_entry else None), add="+")

        self.clear_mask_btn = ctk.CTkButton(right, text="Clear mask (Segment)", command=self._clear_mask)
        self.clear_boxes_btn = ctk.CTkButton(right, text="Clear boxes (Detect)", command=self._clear_boxes)
        self.clear_mask_btn.grid(row=6, column=0, sticky="ew", padx=6, pady=6)
        self.clear_boxes_btn.grid(row=7, column=0, sticky="ew", padx=6, pady=6)

        self.export_btn = ctk.CTkButton(right, text="Export", fg_color="#2a8", hover_color="#277", command=self._export_labels)
        self.export_btn.grid(row=8, column=0, sticky="ew", padx=6, pady=10)

        self.help_label = ctk.CTkLabel(right, justify="left",
            text="ปุ่มลัด:\n- Q: รูปก่อนหน้า\n- E: รูปถัดไป\n- D: คัดลอก Label จากรูปก่อนหน้า\nซูม: Ctrl + Scroll")
        self.help_label.grid(row=9, column=0, sticky="w", padx=6, pady=(6,0))

        # State
        self.current_project_dir = None
        self.images = []  # list of file paths
        self.thumbs = []
        self.thumb_widgets = []
        self.selected_index = -1

        # per-image annotations
        # boxes: list of (cls, x1,y1,x2,y2) in image coordinates
        # mask: PIL single-channel ("L") mask 0/255
        self.boxes_by_image = dict()
        # self.mask_by_image = dict()
        self.mask_by_image = dict()
        self.size_by_image = dict()


        # Viewer state
        self.base_img = None          # PIL
        self.base_img_path = None
        self.zoom_scale = 1.0
        self.min_zoom, self.max_zoom = 0.2, 6.0
        self.tk_img_handle = None
        self.canvas_img_id = None
        self.drawing_box = False
        self.box_start = None

        self.tkcanvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.tkcanvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.tkcanvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.tkcanvas.bind("<Control-MouseWheel>", self.on_mouse_wheel)  # Windows
        self.tkcanvas.bind("<Control-Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.tkcanvas.bind("<Control-Button-5>", self.on_mouse_wheel)    # Linux scroll dn

        # keyboard shortcuts bind to root
        # แก้ไขบรรทัดที่ 422
        # self.bind("<q>", lambda e: self.prev_image())
        # self.bind("<e>", lambda e: self.next_image())
        # self.bind("<E>", lambda e: self.next_image())
        # self.bind("<d>", lambda e: self.copy_prev_labels())
        # self.bind("<D>", lambda e: self.copy_prev_labels())

    def _add_class_from_entry(self):
        raw = self.class_entry.get().strip()
        if not raw:
            return
        # รองรับกรอกได้หลายค่าคั่นด้วย comma หรือ space
        parts = []
        for tok in raw.replace(",", " ").split():
            if tok.strip().isdigit():
                parts.append(int(tok.strip()))
        if not parts:
            messagebox.showerror("คลาสไม่ถูกต้อง", "กรุณากรอกตัวเลข เช่น 0 หรือ 0,1,2")
            return

        last = None
        for cid in parts:
            self._ensure_class_button(cid)
            last = cid

        # ตั้ง active เป็นตัวสุดท้ายที่เพิ่ม
        if last is not None:
            self._set_active_class(last)

        # เคลียร์ช่องและย้ายโฟกัสออก
        self.class_entry.delete(0, "end")
        self.tkcanvas.focus_set()

    def _ensure_class_button(self, cid: int):
        if cid in self.class_buttons:
            return
        btn = ctk.CTkButton(
            self.class_palette,
            text=f"Class {cid}",
            fg_color=self._color_for_class(cid),
            hover_color="#1f1f1f",
            command=lambda c=cid: self._set_active_class(c)
        )
        # วางปุ่มแบบไหลลง (stack) เรียบง่าย
        btn.pack(fill="x", padx=4, pady=3)
        self.class_buttons[cid] = btn
        self._refresh_class_buttons()

    def _set_active_class(self, cid: int):
        self.active_class = cid
        self._refresh_class_buttons()
        # โฟกัสกลับไปที่ canvas เพื่อให้กดคีย์ลัด/ลากได้เลย
        self.tkcanvas.focus_set()

    def _refresh_class_buttons(self):
        for cid, btn in self.class_buttons.items():
            if cid == self.active_class:
                btn.configure(border_width=2, border_color="#fff")
            else:
                btn.configure(border_width=0)

    def _color_for_class(self, cid: int):
        # สีวนตามพาเลตต์
        palette = [
            "#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0",
            "#00BCD4", "#8BC34A", "#FFC107", "#795548", "#FF5722",
        ]
        return palette[cid % len(palette)]

    def handle_key(self, event):
        key = event.keysym.lower()
        if key == "q":
            self.prev_image()
            return "break"
        elif key == "e":
            self.next_image()
            return "break"
        elif key == "d":
            self.copy_prev_labels()
            return "break"

    def _list_project_names(self):
        return sorted([d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))])

    def refresh_projects(self):
        self.project_combo.configure(values=self._list_project_names())

    def _on_project_selected(self, name):
        if not name:
            return
        proj_dir = os.path.join(PROJECTS_DIR, name)
        images_dir = os.path.join(proj_dir, "images")
        if not os.path.isdir(images_dir):
            messagebox.showerror("ไม่พบโฟลเดอร์รูป", f"ไม่มีโฟลเดอร์ images ในโปรเจกต์: {name}")
            return
        self.load_folder(images_dir)

    def _browse_external(self):
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ที่มีรูปภาพ")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        paths.sort()
        if not paths:
            messagebox.showwarning("ไม่มีรูป", "โฟลเดอร์นี้ไม่มีไฟล์รูปที่รองรับ")
            return
        self.current_project_dir = folder
        self.images = paths
        self._populate_thumbs()
        self.selected_index = 0
        self._load_current_image()

    def _populate_thumbs(self):
        for w in self.thumb_widgets:
            w.destroy()
        self.thumb_widgets.clear()
        self.thumbs.clear()
        for i, p in enumerate(self.images):
            pil = Image.open(p).convert("RGB")
            th = pil.copy()
            th.thumbnail((180, 120), Image.LANCZOS)
            ctkimg = pil_to_ctk_image(th)
            btn = ctk.CTkButton(self.thumb_scroll, image=ctkimg, text=os.path.basename(p), compound="top",
                                command=lambda idx=i: self._select_index(idx))
            btn.image = ctkimg
            btn.pack(fill="x", padx=6, pady=4)
            self.thumb_widgets.append(btn)
            self.thumbs.append(th)

    def _select_index(self, idx):
        self.selected_index = idx
        self._load_current_image()

    def _load_current_image(self):

        if self.selected_index < 0 or self.selected_index >= len(self.images):
            return
        path = self.images[self.selected_index]

        self.base_img_path = path
        self.base_img = Image.open(path).convert("RGB")
        self.size_by_image[path] = (self.base_img.width, self.base_img.height)
        # init annotation storage if missing
        self.boxes_by_image.setdefault(path, [])
        if path not in self.mask_by_image:
            self.mask_by_image[path] = {}  # เริ่มด้วย dict ว่าง

        self.zoom_scale = 1.0
        self._render_canvas()

        # highlight selected thumb
        for i, btn in enumerate(self.thumb_widgets):
            if i == self.selected_index:
                btn.configure(fg_color="#2a2a3a")
            else:
                btn.configure(fg_color="transparent")

    def _render_canvas(self):
        if self.base_img is None:
            return
        img = self.base_img

        # --- overlay mask ---
        disp = img.convert("RGBA")
        masks = self.mask_by_image.get(self.base_img_path, {})
        for cid, m in masks.items():
            if m is None:
                continue
            m_np = np.array(m)
            if m_np.max() == 0:
                continue
            color_hex = self._color_for_class(cid).lstrip("#")
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            overlay = Image.new("RGBA", disp.size, (r, g, b, 0))
            ov_np = np.array(overlay)
            ov_np[m_np > 0, 3] = int(255 * 0.30)
            overlay = Image.fromarray(ov_np, "RGBA")
            disp = Image.alpha_composite(disp, overlay)

        disp = disp.convert("RGB")
        disp = draw_boxes_on_pil(disp, self.boxes_by_image.get(self.base_img_path, []),
                                 color=(0, 255, 0), width=2)

        # --- apply zoom ---
        if abs(self.zoom_scale - 1.0) > 1e-3:
            w, h = disp.size
            disp = disp.resize((int(w * self.zoom_scale), int(h * self.zoom_scale)), Image.LANCZOS)

        # --- คำนวณ offset ให้วางภาพกึ่งกลาง ---
        canvas_w = self.tkcanvas.winfo_width()
        canvas_h = self.tkcanvas.winfo_height()
        img_w, img_h = disp.size
        self.img_offset_x = max((canvas_w - img_w) // 2, 0)
        self.img_offset_y = max((canvas_h - img_h) // 2, 0)

        self.tk_img_handle = ImageTk.PhotoImage(disp)
        if self.canvas_img_id is None:
            self.canvas_img_id = self.tkcanvas.create_image(self.img_offset_x, self.img_offset_y,
                                                            image=self.tk_img_handle, anchor="nw")
        else:
            self.tkcanvas.itemconfig(self.canvas_img_id, image=self.tk_img_handle)
            self.tkcanvas.coords(self.canvas_img_id, self.img_offset_x, self.img_offset_y)

        self.tkcanvas.config(scrollregion=(0, 0, img_w + self.img_offset_x, img_h + self.img_offset_y))

    def _img_coords_from_canvas(self, x, y):
        """แปลงพิกัดจาก canvas → พิกัดภาพ"""
        cx = self.tkcanvas.canvasx(x)
        cy = self.tkcanvas.canvasy(y)
        ix = (cx - self.img_offset_x) / self.zoom_scale
        iy = (cy - self.img_offset_y) / self.zoom_scale
        return ix, iy

    def on_mouse_wheel(self, event):
        if self.base_img is None:
            return

        # พิกัดเมาส์บน canvas (รวม scroll แล้ว)
        canvas_x = self.tkcanvas.canvasx(event.x)
        canvas_y = self.tkcanvas.canvasy(event.y)

        # แปลงเป็นพิกัดภาพก่อนซูม (หัก offset ก่อน)
        img_x_before = (canvas_x - self.img_offset_x) / self.zoom_scale
        img_y_before = (canvas_y - self.img_offset_y) / self.zoom_scale

        # ปรับ scale
        old_scale = self.zoom_scale
        if event.delta > 0 or getattr(event, 'num', None) == 4:
            self.zoom_scale = min(self.max_zoom, self.zoom_scale * 1.1)
        else:
            self.zoom_scale = max(self.min_zoom, self.zoom_scale / 1.1)

        if abs(self.zoom_scale - old_scale) < 1e-6:
            return

        # render ใหม่ (จะอัปเดต offset ด้วย)
        self._render_canvas()

        # พิกัดใหม่ของจุดเดิมบน canvas หลังซูม
        new_canvas_x = img_x_before * self.zoom_scale + self.img_offset_x
        new_canvas_y = img_y_before * self.zoom_scale + self.img_offset_y

        # คำนวณ scroll ให้จุดเดิมอยู่ตรงเมาส์
        canvas_width = self.tkcanvas.winfo_width()
        canvas_height = self.tkcanvas.winfo_height()
        bbox = self.tkcanvas.bbox("all")

        if bbox:
            target_x = (new_canvas_x - event.x) / (bbox[2] - canvas_width)
            target_y = (new_canvas_y - event.y) / (bbox[3] - canvas_height)
            self.tkcanvas.xview_moveto(target_x)
            self.tkcanvas.yview_moveto(target_y)

    def on_mouse_down(self, event):
        if self.base_img is None:
            return
        ix, iy = self._img_coords_from_canvas(event.x, event.y)
        tool = self.tool_var.get()
        if tool == "box":
            self.drawing_box = True
            self.box_start = (ix, iy)
        else:
            # draw brush at point
            self._paint_at(ix, iy)

    def on_mouse_drag(self, event):
        if self.base_img is None:
            return
        ix, iy = self._img_coords_from_canvas(event.x, event.y)
        tool = self.tool_var.get()
        if tool == "box" and self.drawing_box:
            # show temp box by updating overlay only at render time (lightweight approach)
            pass
        elif tool == "segment":
            self._paint_at(ix, iy)

    def on_mouse_up(self, event):
        if self.base_img is None:
            return
        tool = self.tool_var.get()
        if tool == "box" and self.drawing_box and self.box_start is not None:
            ix, iy = self._img_coords_from_canvas(event.x, event.y)
            x1, y1 = self.box_start
            x2, y2 = ix, iy
            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                cls_id = self._current_class()
                if cls_id is None:
                    return
                # normalize order
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                self.boxes_by_image[self.base_img_path].append((cls_id, x1, y1, x2, y2))
                self._render_canvas()
            self.drawing_box = False
            self.box_start = None

    def _paint_at(self, ix, iy):
        cls_id = self._current_class()
        if cls_id is None:
            return
        r = int(self.brush_size.get())
        masks = self.mask_by_image.get(self.base_img_path, {})
        mask = masks.get(cls_id)
        if mask is None:
            mask = Image.new("L", (self.base_img.width, self.base_img.height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=255)
        masks[cls_id] = mask
        self.mask_by_image[self.base_img_path] = masks
        self._render_canvas()

    def _current_class(self):

        if self.active_class is None:
            messagebox.showwarning("ยังไม่เลือกคลาส",
                                   "กรุณาเลือกคลาสจากปุ่ม Class ก่อน หรือพิมพ์ตัวเลขแล้วกด Enter เพื่อเพิ่ม")
            return None
        return self.active_class

    def _clear_mask(self):
        if self.base_img_path is None:
            return
        cls_id = self._current_class()
        if cls_id is None:
            return
        masks = self.mask_by_image.get(self.base_img_path, {})
        if cls_id in masks:
            masks[cls_id] = Image.new("L", (self.base_img.width, self.base_img.height), 0)
            self.mask_by_image[self.base_img_path] = masks
            self._render_canvas()

    def _clear_boxes(self):
        if self.base_img_path is None:
            return
        self.boxes_by_image[self.base_img_path] = []
        self._render_canvas()

    def prev_image(self):
        if not self.images:
            return
        self.selected_index = max(0, self.selected_index - 1)
        self._load_current_image()

    def next_image(self):
        if not self.images:
            return
        self.selected_index = min(len(self.images) - 1, self.selected_index + 1)
        self._load_current_image()

    def copy_prev_labels(self):
        if not self.images or self.selected_index <= 0:
            return
        cur = self.images[self.selected_index]
        prev = self.images[self.selected_index - 1]

        # boxes
        prev_boxes = self.boxes_by_image.get(prev, [])
        if prev_boxes:
            pw, ph = self.size_by_image.get(prev, (None, None))
            cw, ch = self.size_by_image.get(cur, (None, None))
            if pw and ph and cw and ch and (pw != cw or ph != ch):
                scale_x = cw / pw;
                scale_y = ch / ph
                new_boxes = []
                for (cls_id, x1, y1, x2, y2) in prev_boxes:
                    new_boxes.append(
                        (cls_id, int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)))
                self.boxes_by_image[cur] = new_boxes
            else:
                self.boxes_by_image[cur] = list(prev_boxes)

        # masks (หลายคลาส)
        prev_masks = self.mask_by_image.get(prev, {})
        if prev_masks:
            cw, ch = self.size_by_image.get(cur, (None, None))
            new_masks = {}
            for cid, pmask in prev_masks.items():
                if pmask is None:
                    continue
                if pmask.size != (cw, ch):
                    new_masks[cid] = pmask.resize((cw, ch), Image.NEAREST)
                else:
                    new_masks[cid] = pmask.copy()
            self.mask_by_image[cur] = new_masks

        self._render_canvas()

    def _export_labels(self):
        if not self.images:
            messagebox.showwarning("ไม่มีรูป", "ยังไม่ได้เลือกโฟลเดอร์รูป")
            return
        out_dir = filedialog.askdirectory(title="เลือกโฟลเดอร์ปลายทางสำหรับ Export")
        if not out_dir:
            return
        images_out = os.path.join(out_dir, "images")
        labels_out = os.path.join(out_dir, "labels")
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)
        count = 0
        for path in self.images:
            pil_img = Image.open(path).convert("RGB")
            w, h = pil_img.size
            base = os.path.splitext(os.path.basename(path))[0]
            # copy/save image
            save_img_path = os.path.join(images_out, base + ".jpg")
            pil_img.save(save_img_path, quality=95)

            # write label file
            label_lines = []
            # boxes
            for (cls_id, x1, y1, x2, y2) in self.boxes_by_image.get(path, []):
                cx, cy, bw, bh = compute_yolo_bbox(w, h, x1, y1, x2, y2)
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # segmentation as polygons (single combined mask)
            # segmentation: บันทึกตามคลาส
            masks = self.mask_by_image.get(path, {})
            for cls_id, mask in masks.items():
                if mask is None:
                    continue
                mask_np = np.array(mask)
                if mask_np.max() == 0:
                    continue
                polys = mask_to_polygons(mask_np)
                for poly in polys:
                    coords = normalize_polygon(poly, w, h)
                    line = f"{cls_id} " + " ".join(coords)
                    label_lines.append(line)

            with open(os.path.join(labels_out, base + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))
            count += 1

        messagebox.showinfo("Export เสร็จสิ้น", f"ส่งออก {count} รูป ไปยัง:\n{out_dir}")


# ---------- Tab 3: Train & Test ----------
class TrainTestTab(ctk.CTkFrame):
    def __init__(self, master, session_logger):
        super().__init__(master)
        self.session_logger = session_logger

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Left: training
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(7, weight=1)

        self.ds_btn = ctk.CTkButton(left, text="เลือกโฟลเดอร์ Dataset (images + labels)", command=self._choose_dataset)
        self.ds_btn.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        mode_frame = ctk.CTkFrame(left)
        mode_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        self.mode_var = ctk.StringVar(value="detect")
        ctk.CTkLabel(mode_frame, text="โหมด:").grid(row=0, column=0, padx=6)
        ctk.CTkRadioButton(mode_frame, text="YOLO Detect", variable=self.mode_var, value="detect").grid(row=0, column=1, padx=6)
        ctk.CTkRadioButton(mode_frame, text="YOLO Segment", variable=self.mode_var, value="segment").grid(row=0, column=2, padx=6)

        model_frame = ctk.CTkFrame(left)
        model_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=6)
        ctk.CTkLabel(model_frame, text="ขนาดโมเดล:").grid(row=0, column=0, padx=6)
        self.model_size = ctk.CTkComboBox(model_frame, values=["n","s","m","l","x"])
        self.model_size.set("n")
        self.model_size.grid(row=0, column=1, padx=6, pady=6)

        params = ctk.CTkFrame(left)
        params.grid(row=3, column=0, sticky="ew", padx=6, pady=6)
        params.grid_columnconfigure(1, weight=1)

        # Parameters
        self.num_classes = ctk.CTkEntry(params, placeholder_text="จำนวน Class (เช่น 2)")
        self.class_names = ctk.CTkEntry(params, placeholder_text='ชื่อ Class (เช่น "OK,NG")')
        self.epochs = ctk.CTkEntry(params, placeholder_text="จำนวน Epoch (เช่น 100)")
        self.batch = ctk.CTkEntry(params, placeholder_text="Batch Size (เช่น 16)")
        self.imgsz = ctk.CTkEntry(params, placeholder_text="Image Size (เช่น 640)")
        self.project_name = ctk.CTkEntry(params, placeholder_text="Project Name")

        items = [
            ("จำนวน Class:", self.num_classes),
            ("ชื่อ Class:", self.class_names),
            ("Epoch:", self.epochs),
            ("Batch Size:", self.batch),
            ("Image Size:", self.imgsz),
            ("Project Name:", self.project_name),
        ]
        for i, (lbl, widget) in enumerate(items):
            ctk.CTkLabel(params, text=lbl).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            widget.grid(row=i, column=1, sticky="ew", padx=6, pady=4)

        self.train_btn = ctk.CTkButton(left, text="เริ่มการฝึก (Train)", fg_color="#2a8", hover_color="#277", command=self._start_train)
        self.train_btn.grid(row=4, column=0, sticky="ew", padx=6, pady=8)

        # Training status
        status = ctk.CTkFrame(left)
        status.grid(row=5, column=0, sticky="ew", padx=6, pady=6)
        status.grid_columnconfigure(0, weight=1)

        self.progress_label = ctk.CTkLabel(status, text="ความคืบหน้า: 0%")
        self.progress_label.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.progress_bar = ctk.CTkProgressBar(status)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        self.eta_label = ctk.CTkLabel(status, text="เหลือเวลา: -")
        self.eta_label.grid(row=2, column=0, sticky="w", padx=6, pady=4)

        self.graph_label = ctk.CTkLabel(left, text="(จะแสดงกราฟเมื่อฝึกเสร็จ)")
        self.graph_label.grid(row=6, column=0, sticky="nsew", padx=6, pady=6)

        # Right: testing
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        # เพิ่มใน __init__() ก่อน self.test_img_label
        self.test_mode_var = ctk.StringVar(value="Mode 1")
        self.test_mode_menu = ctk.CTkOptionMenu(
            right,
            values=["Mode 1", "Mode 2"],
            variable=self.test_mode_var,
            command=self._switch_test_mode
        )
        self.test_mode_menu.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=4)

        self.test_img_label = ctk.CTkLabel(right, text="ตัวอย่างรูปสำหรับทดสอบ", anchor="center")
        self.test_img_label.grid(row=3, column=0, sticky="nsew", padx=6, pady=6)

        self.upload_btn = ctk.CTkButton(right, text="อัปโหลดรูป", command=self._upload_test_image)
        self.model_btn = ctk.CTkButton(right, text="เลือกโมเดล (.pt)", command=self._choose_model)
        self.infer_btn = ctk.CTkButton(right, text="เริ่มการตรวจสอบ", command=self._run_inference)

        self.infer_info = ctk.CTkLabel(right, text="", justify="left")
        self.infer_info.grid(row=5, column=0, sticky="ew", padx=6, pady=6)

        # __init__(): สร้างปุ่ม แต่ไม่ grid
        self.start_cam_btn = ctk.CTkButton(right, text="เริ่มกล้อง", command=self._start_camera)

        self.stop_cam_btn = ctk.CTkButton(right, text="หยุดกล้อง", command=self._stop_camera)
        self._switch_test_mode("Mode 1")


        # State
        self.dataset_dir = None
        self.train_thread = None
        self.monitor_thread = None
        self.stop_monitor = threading.Event()
        self.training_run_dir = None
        self.epoch_times = deque(maxlen=5)
        self.train_start_ts = None

        self.test_image = None
        self.test_image_ctk = None
        self.model_path = None
        self.running_cam = False

    def _switch_test_mode(self, mode):
        # ซ่อนปุ่มทั้งหมดก่อน
        for widget in [self.upload_btn, self.infer_btn, self.start_cam_btn, self.stop_cam_btn, self.model_btn]:
            widget.grid_forget()

        if mode == "Mode 1":
            self.upload_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
            self.model_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
            self.infer_btn.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=8)
        else:
            self.model_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
            self.start_cam_btn.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
            self.stop_cam_btn.grid(row=2, column=1, sticky="ew", padx=6, pady=4)

    def _choose_dataset(self):
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ที่มี images และ labels")
        if not folder:
            return
        # Expect images/ and labels/ inside
        imgs = os.path.join(folder, "images")
        lbls = os.path.join(folder, "labels")
        if not (os.path.isdir(imgs) and os.path.isdir(lbls)):
            messagebox.showerror("โครงสร้างไม่ถูกต้อง", "ต้องมีโฟลเดอร์ images และ labels ภายใน")
            return
        self.dataset_dir = folder
        messagebox.showinfo("เลือก Dataset แล้ว", folder)

    def _build_data_yaml(self, names, nc, imgsz):
        # Split dataset into train/val (80/20) if not existing
        imgs = os.path.join(self.dataset_dir, "images")
        lbls = os.path.join(self.dataset_dir, "labels")

        def list_files(d, ext=(".jpg", ".jpeg", ".png")):
            arr = []
            for e in ext:
                arr.extend(glob.glob(os.path.join(d, f"*{e}")))
            arr.sort()
            return arr

        image_files = list_files(imgs)
        if not image_files:
            raise RuntimeError("ไม่พบไฟล์รูปใน images/")
        # If already has train/val, use them
        if os.path.isdir(os.path.join(imgs, "train")) and os.path.isdir(os.path.join(imgs, "val")):
            data = {
                "path": self.dataset_dir,
                "train": os.path.join("images", "train"),
                "val": os.path.join("images", "val"),
                "nc": nc,
                "names": names
            }
        else:
            # Create split by moving/copying indexes into subfolders
            tr_imgs = os.path.join(imgs, "train"); va_imgs = os.path.join(imgs, "val")
            tr_lbls = os.path.join(lbls, "train"); va_lbls = os.path.join(lbls, "val")
            os.makedirs(tr_imgs, exist_ok=True); os.makedirs(va_imgs, exist_ok=True)
            os.makedirs(tr_lbls, exist_ok=True); os.makedirs(va_lbls, exist_ok=True)

            # Prepare split indexes
            n = len(image_files)
            n_val = max(1, int(0.2 * n))
            val_set = set(image_files[-n_val:])
            for img_path in image_files:
                base = os.path.splitext(os.path.basename(img_path))[0]
                lbl_src = os.path.join(lbls, base + ".txt")
                if img_path in val_set:
                    shutil.copy2(img_path, os.path.join(va_imgs, os.path.basename(img_path)))
                    if os.path.exists(lbl_src):
                        shutil.copy2(lbl_src, os.path.join(va_lbls, os.path.basename(lbl_src)))
                else:
                    shutil.copy2(img_path, os.path.join(tr_imgs, os.path.basename(img_path)))
                    if os.path.exists(lbl_src):
                        shutil.copy2(lbl_src, os.path.join(tr_lbls, os.path.basename(lbl_src)))

            data = {
                "path": self.dataset_dir,
                "train": os.path.join("images", "train"),
                "val": os.path.join("images", "val"),
                "nc": nc,
                "names": names
            }

        yaml_path = os.path.join(self.dataset_dir, f"data_{now_str()}.yaml")
        try:
            import yaml as pyyaml  # If user has pyyaml
            with open(yaml_path, "w", encoding="utf-8") as f:
                pyyaml.safe_dump(data, f, allow_unicode=True)
        except Exception:
            # write plain YAML-like text
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(f"path: {data['path']}\n")
                f.write(f"train: {data['train']}\n")
                f.write(f"val: {data['val']}\n")
                f.write(f"nc: {data['nc']}\n")
                f.write("names: [")
                f.write(", ".join([f"'{n}'" for n in data["names"]]))
                f.write("]\n")
        return yaml_path

    def _start_train(self):
        if YOLO is None:
            messagebox.showerror("ยังไม่พร้อม", "ไม่พบไลบรารี ultralytics กรุณาติดตั้งด้วย: pip install ultralytics")
            return
        if not self.dataset_dir:
            messagebox.showwarning("ยังไม่เลือก Dataset", "กรุณาเลือกโฟลเดอร์ Dataset ก่อน")
            return
        # Reset state
        self.stop_monitor.set()  # หยุด monitor เก่า
        time.sleep(0.5)  # รอให้ thread เก่าหยุด
        self.stop_monitor.clear()
        self.training_run_dir = None
        self.epoch_times.clear()

        try:
            nc = int(self.num_classes.get().strip())
            epochs = int(self.epochs.get().strip())
            batch = int(self.batch.get().strip())
            imgsz = int(self.imgsz.get().strip())
        except Exception:
            messagebox.showerror("พารามิเตอร์ไม่ถูกต้อง", "กรุณากรอกจำนวน class/epochs/batch/imgsz เป็นตัวเลข")
            return
        model_size = self.model_size.get()
        mode = self.mode_var.get()
        names = [s.strip() for s in self.class_names.get().split(",") if s.strip()]
        if len(names) != nc:
            if not messagebox.askyesno("คำเตือน", "จำนวนชื่อ Class ไม่ตรงกับ nc ต้องการดำเนินการต่อหรือไม่?"):
                return
            # Pad or trim names
            if len(names) < nc:
                names += [f"class_{i}" for i in range(len(names), nc)]
            else:
                names = names[:nc]
        project_name = safe_filename(self.project_name.get().strip() or f"{mode}_{model_size}_{now_str()}")

        data_yaml = self._build_data_yaml(names, nc, imgsz)

        # choose model
        model_name = f"model/yolo11{model_size}.pt" if mode == "detect" else f"model/yolo11{model_size}-seg.pt"
        self.train_btn.configure(state="disabled")
        self.stop_monitor.clear()
        self.progress_bar.set(0.0)
        self.progress_label.configure(text="ความคืบหน้า: 0%")
        self.eta_label.configure(text="เหลือเวลา: -")
        self.graph_label.configure(text="(กำลังฝึก...)")
        self.epoch_times.clear()
        self.train_start_ts = time.time()

        def train_worker():
            nonlocal model_name, data_yaml, epochs, batch, imgsz, project_name
            try:
                model = YOLO(model_name)
                results = model.train(
                    data=data_yaml,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    name=project_name,
                    hsv_h=0.05,
                    hsv_s=0.6,
                    hsv_v=0.5,
                    scale=0.8,
                    translate=0.2,
                    fliplr=0.5,
                    flipud=0.1,
                    mosaic=1.0,
                    mixup=0.5,
                    erasing=0.3,
                    lr0=0.0005,  # อัตราการเรียนรู้เริ่มต้น
                    lrf=0.0001,  # อัตราการเรียนรู้สุดท้าย
                    momentum=0.937,  # โมเมนตัมสำหรับ Optimizer
                    weight_decay=0.0005,  # การลดน้ำหนัก (L2 regularization)
                    augment=True,
                    patience=epochs,
                    verbose=False
                )
                del model
                import gc
                gc.collect()
                self._show_training_graph()
                # Find run dir
                # Ultralytics runs dir defaults to 'runs/detect' or 'runs/segment'
                task_dir = "detect" if self.mode_var.get() == "detect" else "segment"
                runs_root = os.path.join(APP_ROOT, "runs", task_dir, project_name)
                latest = sorted(glob.glob(os.path.join(runs_root, "*")), key=os.path.getmtime)
                self.training_run_dir = latest[-1] if latest else None
            except Exception as e:
                messagebox.showerror("Training ล้มเหลว", str(e))
            finally:
                # บอก monitor thread ให้หยุด
                self.stop_monitor.set()
                # คืนปุ่ม Train ให้กดได้
                self.train_btn.configure(state="normal")
                pass

        def monitor_worker():
            # wait for results.csv and update progress
            # scan results.csv within runs/* path
            time.sleep(2.0)
            task_dir = "detect" if self.mode_var.get() == "detect" else "segment"
            # poll until we can find a results.csv
            results_csv = None
            while not self.stop_monitor.is_set():
                # if training_run_dir known, check there first
                if self.training_run_dir:
                    rc = os.path.join(self.training_run_dir, "results.csv")
                    if os.path.isfile(rc):
                        results_csv = rc
                        break
                # else, try to find most recent results.csv
                candidates = glob.glob(os.path.join(APP_ROOT, "runs", task_dir, f"{project_name}", "results.csv"))
                if candidates:
                    results_csv = sorted(candidates, key=os.path.getmtime)[-1]
                    # set training_run_dir based on this file
                    self.training_run_dir = os.path.dirname(results_csv)
                    break
                time.sleep(1.0)

            last_epoch = 0
            total_epochs = 1
            while not self.stop_monitor.is_set() and results_csv and os.path.isfile(results_csv):
                try:
                    with open(results_csv, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last = rows[-1]
                            # 'epoch' column usually present
                            try:
                                cur_epoch = int(float(last.get("epoch", len(rows)-1)))
                            except Exception:
                                cur_epoch = len(rows) - 1
                            # estimate total from user input epochs
                            total_epochs = max(total_epochs, int(self.epochs.get() or "1"))
                            # update timing
                            if cur_epoch > last_epoch:
                                self.epoch_times.append(time.time())
                                last_epoch = cur_epoch
                            prog = min(100, int(100 * (cur_epoch + 1) / max(1, total_epochs)))
                            # ETA estimation
                            if len(self.epoch_times) >= 2:
                                avg_epoch_time = (self.epoch_times[-1] - self.epoch_times[0]) / (len(self.epoch_times) - 1)
                                remaining = max(0, (total_epochs - (cur_epoch + 1)) * avg_epoch_time)
                            else:
                                remaining = 0
                            hrs = int(remaining // 3600)
                            mins = int((remaining % 3600) // 60)
                            secs = int(remaining % 60)
                            self.progress_bar.set(prog / 100.0)
                            self.progress_label.configure(text=f"ความคืบหน้า: {prog}%  {cur_epoch}/{total_epochs} epoch")
                            self.eta_label.configure(text=f"เหลือเวลา ~ {hrs} ชั่วโมง {mins} นาที {secs} วินาที")

                except Exception:
                    pass
                time.sleep(1.0)
            self.stop_monitor.set()
            self.train_btn.configure(state="normal")

            # On finish: show graph
            self._show_training_graph()
            print("=========================================================")
            # re-enable train button
            self.train_btn.configure(state="normal")
            # log session
            try:
                end_ts = time.time()
                self.session_logger.add_train(
                    mode=self.mode_var.get(),
                    model_size=self.model_size.get(),
                    num_classes=int(self.num_classes.get() or "0"),
                    class_names=[s.strip() for s in self.class_names.get().split(",") if s.strip()],
                    epochs=int(self.epochs.get() or "0"),
                    batch=int(self.batch.get() or "0"),
                    imgsz=int(self.imgsz.get() or "0"),
                    project_name=self.project_name.get().strip(),
                    start_time=self.train_start_ts or end_ts,
                    end_time=end_ts
                )
            except Exception:
                pass

        self.training_run_dir = None
        self.stop_monitor.clear()

        self.train_thread = threading.Thread(target=train_worker, daemon=True)
        self.train_thread.start()
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        # self._show_training_graph()

    def _show_training_graph(self):
        # Try to display results.png, else generate from results.csv
        if not self.training_run_dir:
            self.graph_label.configure(text="ไม่พบผลลัพธ์การฝึก")
            return
        res_png = os.path.join(self.training_run_dir, "results.png")
        if os.path.isfile(res_png):
            try:
                pil = Image.open(res_png).convert("RGB")
                ctkimg = pil_to_ctk_image(pil, size=(800, 400))
                self.graph_label.configure(image=ctkimg, text="")
                self.graph_label.image = ctkimg
                return
            except Exception:
                pass
        # Try from CSV
        res_csv = os.path.join(self.training_run_dir, "results.csv")
        if os.path.isfile(res_csv):
            try:
                xs = []
                mAP50 = []
                box_loss = []
                with open(res_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        xs.append(int(float(r.get("epoch", len(xs)))))
                        if "metrics/mAP50(B)" in r:
                            mAP50.append(float(r["metrics/mAP50(B)"] or 0.0))
                        elif "metrics/mAP50(M)" in r:
                            mAP50.append(float(r["metrics/mAP50(M)"] or 0.0))
                        else:
                            mAP50.append(0.0)
                        if "train/box_loss" in r:
                            box_loss.append(float(r["train/box_loss"] or 0.0))
                        else:
                            box_loss.append(0.0)
                plt.figure(figsize=(8,3))
                plt.plot(xs, mAP50, label="mAP50")
                plt.plot(xs, box_loss, label="box_loss")
                plt.legend()
                plt.tight_layout()
                out = os.path.join(self.training_run_dir, "quick_plot.png")
                plt.savefig(out)
                plt.close()
                pil = Image.open(out).convert("RGB")
                ctkimg = pil_to_ctk_image(pil, size=(800, 300))
                self.graph_label.configure(image=ctkimg, text="")
                self.graph_label.image = ctkimg
            except Exception as e:
                self.graph_label.configure(text=f"ไม่สามารถแสดงกราฟได้: {e}")

    def _upload_test_image(self):
        path = filedialog.askopenfilename(title="เลือกภาพสำหรับทดสอบ",
                                          filetypes=[("Images","*.jpg;*.jpeg;*.png;*.bmp")])
        if not path:
            return
        pil = Image.open(path).convert("RGB")
        self.test_image = pil
        self.test_image_ctk = pil_to_ctk_image(pil, size=(800, 500))
        self.test_img_label.configure(image=self.test_image_ctk, text="")
        self.test_img_label.image = self.test_image_ctk

    def _choose_model(self):
        path = filedialog.askopenfilename(title="เลือกไฟล์โมเดล",
                                          filetypes=[("PyTorch Weights","*.pt")])
        if path:
            self.model_path = path
            messagebox.showinfo("เลือกโมเดลแล้ว", path)

    def _start_camera(self):
        # ถ้ามีการเปิดกล้องอยู่แล้ว ให้ปิดก่อน
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        if not self.model_path:
            messagebox.showwarning("ยังไม่เลือกโมเดล", "กรุณาเลือกไฟล์โมเดล .pt ก่อน")
            return

        # เปิดกล้องใหม่
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not self.cap.isOpened():
            messagebox.showerror("กล้องไม่พร้อม", "ไม่สามารถเปิดกล้องได้")
            return

        self.running_cam = True
        self.model = YOLO(self.model_path)
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        while self.running_cam:
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model(frame, conf=0.25, verbose=False)
            plotted = results[0].plot()
            img_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            self.test_image_ctk = pil_to_ctk_image(pil, size=(800, 500))
            self.test_img_label.configure(image=self.test_image_ctk, text="")
            self.test_img_label.image = self.test_image_ctk
            infos = []
            if hasattr(results[0], "boxes") and results[0].boxes is not None:
                confs = results[0].boxes.conf.cpu().numpy().tolist()
                if confs:
                    avg_conf = sum(confs) / len(confs)
                    infos.append(f"จำนวน Detection: {len(confs)} เฉลี่ยความมั่นใจ: {avg_conf:.3f}")
            self.infer_info.configure(text="\n".join(infos))

        self.cap.release()

    def _stop_camera(self):
        self.running_cam = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def _run_inference(self):
        if YOLO is None:
            messagebox.showerror("ยังไม่พร้อม", "ไม่พบไลบรารี ultralytics กรุณาติดตั้งด้วย: pip install ultralytics")
            return
        if self.test_image is None:
            messagebox.showwarning("ยังไม่เลือกรูป", "กรุณาอัปโหลดรูปสำหรับทดสอบ")
            return
        if not self.model_path:
            messagebox.showwarning("ยังไม่เลือกโมเดล", "กรุณาเลือกไฟล์โมเดล .pt")
            return

        def worker():
            try:
                model = YOLO(self.model_path)
                # Predict
                res = model.predict(self.test_image, conf=0.25, verbose=False)
                if not res:
                    self.infer_info.configure(text="ไม่พบผลการตรวจจับ")
                    return
                r = res[0]
                # Plot
                plotted = r.plot()  # numpy array BGR
                pil = Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
                self.test_image_ctk = pil_to_ctk_image(pil, size=(800, 500))
                self.test_img_label.configure(image=self.test_image_ctk, text="")
                self.test_img_label.image = self.test_image_ctk
                # Info
                infos = []
                if hasattr(r, "boxes") and r.boxes is not None:
                    confs = r.boxes.conf.cpu().numpy().tolist() if hasattr(r.boxes, "conf") else []
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist() if hasattr(r.boxes, "cls") else []
                    if confs:
                        avg_conf = sum(confs)/len(confs)
                        infos.append(f"จำนวน Detection: {len(confs)} เฉลี่ยความมั่นใจ: {avg_conf:.3f}")
                    if cls_ids:
                        counts = defaultdict(int)
                        for c in cls_ids: counts[c]+=1
                        infos.append("ต่อคลาส: " + ", ".join([f"{k}: {v}" for k,v in counts.items()]))
                self.infer_info.configure(text="\n".join(infos) if infos else "ไม่มีข้อมูลสถิติ")
            except Exception as e:
                messagebox.showerror("ทดสอบไม่สำเร็จ", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def destroy(self):
        try:
            self.stop_monitor.set()
            self.running_cam = False
        except Exception:
            pass
        super().destroy()

# ------------ camera_manager ----------------



# ---------- Main App ----------
class YOLOManagerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Project Manager")
        # Fullscreen/responsive
        self.state("zoomed") if sys.platform.startswith("win") else self.attributes("-zoomed", True) if sys.platform == "linux" else self.attributes("-fullscreen", True)
        self.minsize(1200, 800)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.session_logger = SessionLogger()
        self.session_logger._session_start = datetime.now().isoformat()

        # Edge-like top tab bar -> CTkTabview with top placement
        self.tabview = ctk.CTkTabview(self, segmented_button_selected_color="#2a6", segmented_button_unselected_color="#333",
                                      segmented_button_selected_hover_color="#284", segmented_button_unselected_hover_color="#444")

        self.tabview.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab1 = self.tabview.add("📸 ถ่ายภาพ & รูปภาพ")
        self.tab2 = self.tabview.add("✍️ Labeling สำหรับ YOLO")
        self.tab3 = self.tabview.add("🤖 ฝึก & ทดสอบ")

        # Build tabs
        self.projects_index = {}  # name -> path

        def on_saved(name, path):
            self.projects_index[name] = path
            self.label_tab.refresh_projects()

        self.capture_tab = CaptureTab(self.tab1, session_logger=self.session_logger, on_project_saved=on_saved)
        self.capture_tab.pack(fill="both", expand=True)

        def get_projects():
            return self.projects_index

        self.label_tab = LabelTab(self.tab2, get_projects_callable=get_projects)
        self.label_tab.pack(fill="both", expand=True)
        self.bind_all("<Key>", self._global_key_handler)



        self.train_tab = TrainTestTab(self.tab3, session_logger=self.session_logger)
        self.train_tab.pack(fill="both", expand=True)

        # Hook close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _global_key_handler(self, event):
        # ส่งต่อให้ label_tab ถ้าแท็บปัจจุบันคือ Labeling
        if self.tabview.get() == "✍️ Labeling สำหรับ YOLO":
            self.label_tab.handle_key(event)


    def on_close(self):
        # Write session log
        log_path = self.session_logger.write_on_exit()
        try:
            self.capture_tab.destroy()
            self.label_tab.destroy()
            self.train_tab.destroy()
        except Exception:
            pass
        print(f"บันทึกการใช้งาน: {log_path}")
        self.destroy()


if __name__ == "__main__":
    app = YOLOManagerApp()
    app.mainloop()
