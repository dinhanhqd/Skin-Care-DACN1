 # import các gói thư viện cần thiết
from tensorflow.keras.utils import img_to_array

class ImageToArrayPreprocessor:  # Tạo lớp để chuyển ảnh --> mảng
    def __init__(self, dataFormat=None):
        # Lưu ảnh đã được định dạng
        self.dataFormat = dataFormat

    def preprocess(self, image): # Định nghĩa phương thức preprocess trả về mảng
        # Hàm img_to_array của Keras
        return img_to_array(image, data_format=self.dataFormat)
