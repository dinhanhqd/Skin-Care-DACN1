# import the necessary packages
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor
from datasets import simpledatasetloader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2
import argparse

#  Load_model.py -d <folder chứa ảnh phân loại>
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="Nhập vào folder để phân loại")
args = vars(ap.parse_args(["-d", "image"]))

# Khởi tạo danh sách nhãn
classLabels = ["DaThuong", "NotRuoi", "BenhSoi","U_Lanh","UngThu"]

#Lấy danh sách các hình ảnh trong tập dữ liệu sau đó lấy mẫu ngẫu nhiên
# ảnh theo chỉ số để đưa vào đường dẫn hình ảnh
print("[INFO] Đang nạp ảnh mẫu để phân lớp (dự đoán)...")
imagePaths = np.array(list(paths.list_images(args["dataset"]))) #xác định số file trong dataset
idxs = range(0, len(imagePaths)) # Lấy tất cả các chỉ số idxs của ảnh
imagePaths = imagePaths[idxs]

sp = simplepreprocessor.SimplePreprocessor(64, 64) # Thiết lập kích thước ảnh 64 x 64
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng


# Nạp dataset từ đĩa rồi co dãn mức xám của pixel trong vùng [0,1]
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# Nạp model đã pre-trained
print("[INFO] Nạp model mạng pre-trained ...")
model = load_model("model.h5")

# Dự đoán
print("[INFO] Đang dự đoán để phân lớp...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# Lặp qua tất cả các file ảnh trong imagePaths
# Nạp ảnh ví dụ --> Vẽ dự đoán --> Hiển thị ảnh
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)







