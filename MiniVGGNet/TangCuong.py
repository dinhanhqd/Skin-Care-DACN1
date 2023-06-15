import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array
from tensorflow.keras.utils import load_img
from imutils import paths
from PIL import Image

# Đường dẫn thư mục chứa ảnh gốc
image_directory = './datasets/dataSkin/UngThu'

# Đường dẫn thư mục lưu trữ ảnh tăng cường
output_directory = './datasets/dataSkin/UngThu'

# Tiền tố cho các tệp ảnh tăng cường
prefix = 'augmented_'

# Số lần tăng cường ảnh
augmentation_count = 4

# Khởi tạo bộ sinh tăng cường dữ liệu
data_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Lấy danh sách đường dẫn tới tất cả các file ảnh trong thư mục "image"
image_paths = list(paths.list_images(image_directory))

# Lặp qua từng đường dẫn ảnh
for image_path in image_paths:
    # Nạp ảnh đầu vào, chuyển đổi thành mảng NumPy, và thay đổi kích thước
    print(f"Nạp ảnh: {image_path}")
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Tạo bộ sinh ảnh tăng cường và khởi tạo tổng số ảnh được sinh ra
    augmented_images = data_augmentation.flow(image, batch_size=1)

    # Lặp qua số lần tăng cường ảnh
    for i in range(augmentation_count):
        augmented_image = next(augmented_images)
        augmented_image = augmented_image.reshape(image.shape[1:])  # Chuyển đổi kích thước ảnh ban đầu
        augmented_image = (augmented_image * 255).astype(np.uint8)  # Chuyển đổi về kiểu dữ liệu uint8
        output_path = os.path.join(output_directory, f"{prefix}{i}_{os.path.basename(image_path)}")
        img = Image.fromarray(augmented_image)
        img.save(output_path)
        print(f"Lưu ảnh tăng cường: {output_path}")

print("Hoàn thành tăng cường ảnh")
