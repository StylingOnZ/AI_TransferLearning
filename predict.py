import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import sys

MODEL_PATH = "model.h5"

# ============================
# 1. LOAD MODEL
# ============================
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy file model: {MODEL_PATH}")
        print("➡ Hãy chắc chắn rằng model.h5 nằm cùng thư mục predict.py")
        sys.exit()

    print("📦 Đang tải model, vui lòng chờ...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print("❌ Lỗi khi load model!")
        print(e)
        sys.exit()

    print("✅ Model đã load thành công!\n")
    return model


model = load_model_safe()

# Danh sách nhãn ứng với model softmax
CLASS_NAMES = ["cats", "dogs"]


# ============================
# 2. HÀM DỰ ĐOÁN
# ============================
def predict_image(img_path):

    # Kiểm tra file
    if not os.path.exists(img_path):
        print(f"❌ Ảnh không tồn tại: {img_path}")
        return

    # Load ảnh
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception:
        print("❌ Không thể đọc ảnh. Có thể ảnh bị lỗi hoặc sai định dạng!")
        return

    # Tiền xử lý
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    # Dự đoán
    preds = model.predict(img_array)[0]

    # Lấy index class có xác suất cao nhất
    class_id = np.argmax(preds)
    confidence = preds[class_id]

    # ============================
    # IN RA KẾT QUẢ ĐẸP
    # ============================

    print("=======================================")
    print("📸 ẢNH ĐẦU VÀO:", img_path)
    print("---------------------------------------")
    print("📊 XÁC SUẤT DỰ ĐOÁN:")
    print(f"   🐱 Cat : {preds[0]*100:.2f}%")
    print(f"   🐶 Dog : {preds[1]*100:.2f}%")
    print("---------------------------------------")
    print(f"👉 KẾT LUẬN: Đây là **{CLASS_NAMES[class_id].upper()}** "
          f"(Độ tin cậy: {confidence*100:.2f}%)")
    print("=======================================\n")


# ============================
# 3. CHẠY TỪ TERMINAL
# ============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Bạn chưa nhập đường dẫn ảnh!")
        print("📌 Cách dùng:")
        print("   python predict.py dataset/test/cats/0.jpg")
        sys.exit()

    input_path = sys.argv[1]
    predict_image(input_path)
