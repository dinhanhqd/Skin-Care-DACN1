from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Khởi tạo mô hình, shape ảnh đầu vào và số kênh của ảnh đầu vào
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1  # chỉ số của số kênh ảnh đầu vào
                          # giá trị -1 ý muốn nói chỉ số kênh nằm cuối cùng
                          # của danh sách chứa dữ liệu ảnh đầu vào

        # sử dụng 'channels_first' để cập nhật shape và số kênh ảnh đầu vào
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dim = 1

        # Chuỗi layer đầu tiên  CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Chuỗi layer thứ hai CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))# loại bỏ ngẫu nhiên 25% noron dể tránh overfix

        # Thiết lập FC(Fully Connected)  thứ nhất => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))   # Dropout 50%

        # THiết lập FC thứ hai => Hàm phân lớp Softmax
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # Trả về kiến trúc mạng/mô hình
        return model
