# import the necessary packages
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # Lưu ảnh tiền xử lý
        self.preprocessors = preprocessors

        # Nếu bước tiền xử lý là None thì khởi tạo danh sách rỗng
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # Khởi tạo danh sách các đặc trưng và nhãn
        data = []
        labels = []

        # Lặp qua tất cả ảnh đầu vào
        for (i, imagePath) in enumerate(imagePaths):
            # Nạp ảnh và trích xuất nhãn từ đường dẫn định dạng
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # Lặp qua tất cả tiền xử lý và áp dụng cho mỗi ảnh
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # Mỗi ảnh được xử lý là vector đặc trưng bằng cách
            # cập nhật danh sách dữ liệu cùng với nhãn
            data.append(image)
            labels.append(label)

            # Hiển thị ảnh cập nhật
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
               print("[INFO] Đã xử lý {}/{}".format(i + 1,len(imagePaths)))
                # Trả về dữ liệu kiểu tuple gồm dữ liệu và nhãn
        return (np.array(data), np.array(labels))

