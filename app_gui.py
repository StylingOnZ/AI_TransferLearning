import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import cv2
import os

MODEL_PATH = "model.h5"

# ===============================
# 1. LOAD MODEL
# ===============================
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

labels = ["cats", "dogs"]

# ===============================
# 2. GUI WINDOW
# ===============================
window = tk.Tk()
window.title("🐶🐱 AI Dog - Cat Classifier")
window.geometry("700x600")
window.resizable(False, False)

image_label = Label(window)
image_label.pack(pady=10)

result_label = Label(window, text="", font=("Arial", 14), fg="blue")
result_label.pack(pady=10)

percent_label = Label(window, text="", font=("Arial", 12))
percent_label.pack(pady=5)

selected_image_path = None


# ===============================
# 3. HÀM CHỌN ẢNH
# ===============================
def browse_image():
    global selected_image_path

    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.jfif")]
    )

    if not file_path:
        return

    selected_image_path = file_path

    # Hiển thị ảnh trên GUI
    img = Image.open(file_path)
    img = img.resize((350, 350))
    img_tk = ImageTk.PhotoImage(img)

    image_label.configure(image=img_tk)
    image_label.image = img_tk

    result_label.config(text="")
    percent_label.config(text="")


# ===============================
# 4. HÀM DỰ ĐOÁN
# ===============================
def predict_image():
    global selected_image_path

    if not selected_image_path:
        result_label.config(text="❌ Bạn chưa chọn ảnh!")
        return

    # Load image
    img = Image.open(selected_image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0]

    cat_p = pred[0] * 100
    dog_p = pred[1] * 100

    # Kết quả
    if cat_p > dog_p:
        final = f"🐱 Đây là **MÈO** (độ tin cậy: {cat_p:.2f}%)"
    else:
        final = f"🐶 Đây là **CHÓ** (độ tin cậy: {dog_p:.2f}%)"

    # In ra GUI
    result_label.config(text=final)

    percent_label.config(
        text=f"📊 Xác suất:\n"
             f" - Cat: {cat_p:.2f}%\n"
             f" - Dog: {dog_p:.2f}%"
    )


# ===============================
# 5. BUTTONS
# ===============================
btn_browse = Button(window, text="📁 Chọn ảnh", font=("Arial", 12),
                    command=browse_image, width=15, bg="#4CAF50", fg="white")
btn_browse.pack(pady=10)

btn_predict = Button(window, text="🤖 Dự đoán", font=("Arial", 12),
                     command=predict_image, width=15, bg="#2196F3", fg="white")
btn_predict.pack(pady=5)


# ===============================
# 6. START APP
# ===============================
window.mainloop()
