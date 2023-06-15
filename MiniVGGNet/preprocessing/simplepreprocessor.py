# import the necessary packages
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # Lưu image width, height và interpolation
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # Trả về ảnh có kích thước đã thay đổi
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
