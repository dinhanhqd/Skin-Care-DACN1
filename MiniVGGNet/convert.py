import tensorflow as tf

# Load model như bình thường
model = tf.keras.models.load_model('model.h5')

# Khởi tạo một bộ converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Thực hiện convert
tflite_model = converter.convert()

# Write vào file
open("model.tflite", "wb").write(tflite_model)