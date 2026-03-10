import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
from PIL import Image
Image.register_extension("JPEG", ".jfif")

# ============================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ============================
train_dir = r"E:\Study\XuLyAnh\AI_TransferLearning\dataset\train"
test_dir  = r"E:\Study\XuLyAnh\AI_TransferLearning\dataset\test"

# ============================
# 2. KIỂM TRA THƯ MỤC + SỐ ẢNH
# ============================
def check_folder(path):
    if not os.path.exists(path):
        raise Exception(f"❌ Folder không tồn tại: {path}")

    classes = [c for c in os.listdir(path)
               if os.path.isdir(os.path.join(path, c))]

    if len(classes) < 1:
        raise Exception(f"❌ Trong thư mục '{path}' không có class con!")

    total_images = 0
    print(f"\n📂 Kiểm tra thư mục: {path}")
    print("------------------------------------")

    for cls in classes:
        cls_path = os.path.join(path, cls)
        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('jpg','png','jpeg'))])

        print(f" Class '{cls}' có {count} ảnh")
        total_images += count

    if total_images == 0:
        raise Exception(f"❌ Không tìm thấy ảnh nào trong thư mục {path}")

    print(f"✅ Tổng ảnh: {total_images}")

check_folder(train_dir)
check_folder(test_dir)

# ============================
# 3. IMAGE GENERATOR
# ============================
train_gen = ImageDataGenerator(rescale=1/255.)
test_gen = ImageDataGenerator(rescale=1/255.)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical"
)

# ============================
# 4. MODEL
# ============================
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
out = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================
# 5. TRAIN
# ============================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# ============================
# 6. SAVE
# ============================
model.save("model.h5")
print("\n🎉 Đã train xong và lưu model thành công!")
