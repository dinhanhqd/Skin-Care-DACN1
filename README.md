# DoAn_CN1
Chương 5-3. Phân lớp ảnh
(Image Classification)
TS. PHẠM NGUYỄN MINH NHỰT
Email: pnmnhut@vku.udn.vn
Phone: 0903.501.421
• Tăng kích thước mô hình (Tăng số lớp tích chập) càng nhiều tham số mô hình sẽ học
được nhiều
• Thay đổi activation function ( ví dụ: Tanh, ReLU, sigmoid, LeakyReLU…)
• Thay đổi các tham số: tốc độ học, kích thước ảnh đầu vào, số lương epoch…
• Thay đổi thuật toán tối ưu (Adam, SGD, RMSprop…)
• Tăng số lượng dữ liệu train  Nếu không đủ sử dụng kỹ thuật tăng cường dữ liệu (Data
Augmentation)
• Kích thước Mini - batch nhỏ cũng ảnh hưởng độ chính xác  kích thước thường 32
• Thay đổi kiến trúc mô hình CNN…
2
7. Tăng độ chính xác của model
Hình gốc Lật Xoay
Crop ngẫu nhiên
Dịch chuyển màu
Thêm nhiễu Mất mát thông tin Thay đổi độ tương phản
• Khi có ít dữ liệu cho việc training  Data Augmentation là kỹ thuật tạo ra dữ liệu để
training từ dữ liệu đang có
8. Tăng cường dữ liệu (Data Augmentation)
• Sử dụng lệnh tăng cường dữ liệu trong Keras
8. Tăng cường dữ liệu (Data Augmentation)
Data_Aug = ImageDataGenerator(
 rescale=1./255, #Thay đổi tỷ lệ
 rotation_range=40, #Xoay
 width_shift_range=0.2, # Thay đổi chiều rộng
 height_shift_range=0.2, # Thay đổi chiều cao
 shear_range=0.2, # Xén ảnh
 zoom_range=0.2, # Zoom ảnh
 horizontal_flip=True, # Lật ảnh
 fill_mode='nearest')
• Áp dụng để tạo thêm Dataset ảnh hoặc dùng trực tiếp trong chương trình khi train
8. Tăng cường dữ liệu – thực hành
9. Thực hành phân loại ảnh – MiniVGGNet
• VGGNet là CNN sử dụng đối với tập dữ liệu ảnh quy mô lớn
• Được giới thiệu bởi Simonyan và Zisserman vào năm 2014
• VGGNet sử dụng kernel kích thước 3 × 3 trong toàn bộ kiến trúc.
• VGGNet có nhiều phiên bản
• VGGNet16  sử dụng 16 layer
• VGGNet19  sử dụng 19 layer
• …
6
• MiniVGGNet là một phiên bản nhỏ hơn của VGGNetđược  có thể dễ dàng triển khai
trên máy tính hạn chế tài nguyên
• Kiến trúc MiniVGGNet:
• Gồm 02 chuỗi lớp: CONV => RELU => CONV => RELU => POOL
• Theo sau bởi FC => RELU => FC => SOFTMAX
• Hai lớp CONV đầu tiên sẽ học thông qua 32 bộ lọc, mỗi bộ lọc có kích thước 3 × 3
• Hai lớp CONV thứ hai sẽ học thông qua 64 bộ lọc, mỗi bộ lọc có kích thước 3 × 3
• Các lớp POOL sử dụng Max Pooling với cửa sổ kích thước 2 × 2 và Stride (trượt) là 2 × 2
• Sử dụng lớp Batch Normalization sau khi activations  tránh overfitting và tăng độ chính xác
• Lớp Dropout thực hiện sau khi POOL và lớp FC
•  Dataset tham khảo từ: https://www.kaggle.com/alxmamaev/flowers-recognition
7
9. Thực hành phân loại ảnh – MiniVGGNet
• Cấu trúc MiniVGGNet
8
9. Thực hành phân loại ảnh – MiniVGGNet
• Tổ chức project
9
Chứa Dataset cho Trainning, Validation, Testing
05 class
Định nghĩa phương thức nạp dữ liệu
Định nghĩa model và layers
Định nghĩa phương thức chuẩn bị dữ liệu trước khi
nạp vào model để train
Nạp model (*.hdf5) để viết ứng dụng
Train và đánh giá model
Chứa ảnh để thử nghiệm model
9. Thực hành phân loại ảnh – MiniVGGNet
Kiểm tra
11
Đề bài
12
• Tham khảo MiniVGGNet để tạo model phân lớp animals, với yêu cầu:
• Dữ liệu animals: (cat,dog, panda) = (1000,1000,1000)
• Sử dụng kỹ thuật tăng cường dữ liệu, sao cho (cat,dog, panda) = (1500,1500,1500)
• Cố định số epoch=60
• Có thể hiệu chỉnh cấu trúc model, thay đổi tốc độ học…
• Thứ 5 kiểm tra: 13h00 học thì 13h30 báo cáo kết quả  train trước khi vào học
• Điểm:
Độ chính xác < 70% 71% > 71%
<= 75%
>75%
<=80%
>80%
<=90%
>90%
Điểm 0 5 8 8.5 9 10
• Học chuyển tiếp (transfer learning) khả năng sử dụng model được đào tạo trước (pretrained) để tiếp tục học từ dữ liệu mới (dữ liệu mới này chưa được dùng để train cho
model trước đó).
• Ví dụ:
• Trước đây đã tạo mạng CNN để nhận diện chó, mèo và gấu trong ảnh
• Sử dụng Học chuyển tiếp là dùng mạng CNN này để tiếp tục train nhằm nhận diện gấu
xám, gấu bắc cực và gấu trúc khổng lồ (các dữ liệu này chưa được xử dụng để train
cho model trước đó)
• Có 2 hình thức học chuyển tiếp khi áp dụng cho deep learning trong thị giác máy tính:
• Phương pháp 1: Thay đổi mục đích của mạng CNN đã có  để trích xuất đặt trưng.
Phương pháp này gọi là Trích xuất đặc trưng với mạng Pre-Trained CNN
• Phương pháp 2: Thay thế các lớp FC đã có của mạng CNN bằng các lớp FC mới (gắn
vào vùng Top của CNN) và tinh chỉnh (fine-tuning) các trọng số của CNN. Phương pháp
này gọi là Tinh chỉnh (fine-tuning) các trọng số của CNN
13
10. Học chuyển tiếp (transfer learning)
• Xét một pre-trained CNN (VGG16) cho bài toán phân lớp ảnh đã được train trước đó
• Loại bỏ lớp FC và không sử dụng hàm phân lớp (softmax) ở cuối mạng
14
Xe ô tô
4 bánh
Các đặc
trưng
10. Học chuyển tiếp - Phương pháp 1: Trích xuất đặc trưng từ Pre-trained CNN
•  Lớp cuối cùng trong mạng có 512 bộ lọc mỗi bộ có kích thước 7 × 7.
•  Có véc tơ với 7 × 7 × 512 = 25.088 phần tử  véc tơ đặc trưng biểu diễn nội dung của
ảnh.
• Từ đó sử dụng các đặc trưng này để train các mô hình khác  tạo ra các bộ phân lớp để
nhận dạng các lớp ảnh mới khác với độ chính xác cao hơn
15
10. Học chuyển tiếp - Phương pháp 1: Trích xuất đặc trưng từ Pre-trained CNN
16
• Các bước thực hiện theo Phương pháp 1: Trích xuất đặc trưng và train
Trích xuất đặc trưng từ mô
hình Pre -Trained và lưu trữ
(*.hdf5) 
Train từ các đặc trưng và
lưu trữ model mới (*. Pickle)
Ứng dụng model mới
(*. Pickle)
10. Học chuyển tiếp - Phương pháp 1: Trích xuất đặc trưng từ Pre-trained CNN
17
Các lớp
của
kiến trúc
gốc
Các lớp
FC mới
• Xóa phần đầu khỏi mạng, giống
như trong trích xuất đặc trưng
• Xây dựng lớp FC mới
• Đặt lớp FC này trên đầu của kiến
trúc gốc
• Train model với lớp FC mới này
10. Học chuyển tiếp - Phương pháp 2: Tinh chỉnh (fine-tuning) các trọng số của CNN
• Thông thường trước khi train,
đóng băng các lớp của mô hình
gốc (gọi là lớp Base)
• Train lớp FC đến trạng thái “ấm
lên”
• Sau đó, mở đóng băng và tinh
chỉnh (train) các tham số của
mạng
Các lớp bị
đóng băng
Chỉ train lớp
FC để các lớp
chuyển trạng
thái “ấm lên”
Mở đóng băng và
train tất cả
10. Học chuyển tiếp - Phương pháp 2: Tinh chỉnh (fine-tuning) các trọng số của CNN
• Thực hành: Sử dụng phương pháp tinh chỉnh (Fine-Tuning) đối với model VGG16
• Chú ý: Sử dụng kỹ thuật tăng cường ảnh bằng dòng lệnh của Keras
10. Học chuyển tiếp - Phương pháp 2: Tinh chỉnh (fine-tuning) các trọng số của CNN
aug = ImageDataGenerator(
rotation_range=30,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest') 
• VGG16 là mạng CNN được đề xuất bởi K. Simonyan and A. Zisserman, University of
Oxford.
• Model sau khi train bởi mạng VGG16 đạt độ chính xác 92.7%
• Dữ liệu ImageNet gồm 14 triệu hình ảnh thuộc 1000 lớp khác nhau.
10. Học chuyển tiếp - Phương pháp 2: Tinh chỉnh (fine-tuning) các trọng số của CNN
11. Triển khai Model Deep Learning trên Colab
• Colaboratory hay còn gọi là Google Colab là một sản phẩm từ Google Research
• Cho phép chạy các code viết bằng python thông qua trình duyệt
• Ứng dụng các bài toán Data Analysis, Machine Learning, Deep Learning...
• Cung cấp tài nguyên máy tính  GPU và TPU
• Đối với dịch vụ miễn phí giới hạn thời gian sử dụng, tối đa lên tới 12 giờ  Do vậy, kết
hợp Drive Google để lưu tài nguyên 
21
• Bước 1: Đăng nhập Drive Google 
• Bước 2: Sao chép cấu trúc dự án
(trên máy cá nhân) lên Drive Google
• Bước 3: Chọn Drive của tôi  Ứng
dụng khác  Google Colaboratory
22
11. Triển khai Model Deep Learning trên Colab
• Bước 4: Đặt tên file: Chọn tại
Untitled1.ipynb và nhập tên file
mới
• Bước 5: Chọn cấu hình máy tính
trên Colab: Chọn Thời gian chạy 
Thay đổi loại thời gian chạy 
Chọn loại CPU,GPU hoặc TPU
23
11. Triển khai Model Deep Learning trên Colab
• Bước 6: Thực hiện kết nối đến với cloud của Google  nhấn Kết nối
24
11. Triển khai Model Deep Learning trên Colab
• Bước 7: Kết nối Colab với Drive Google  Nhập 02 lệnh
• Bước 8: Nhấn nút thực thi lệnh ( hình Tam giác)  Nhấn link 
 
25
11. Triển khai Model Deep Learning trên Colab
• Bước 8: Đăng nhập mail google
 
26
11. Triển khai Model Deep Learning trên Colab
• Bước 9: Sao chép mã xác thực
• Bước 10: Dán mã vào ô Enter your authorization Code và nhấn enter
27
11. Triển khai Model Deep Learning trên Colab
• Thêm mã: Nhấn vào + Mã
• Thực thi : Nhấn nút Tam giác
• Chuyển vào folder: %cd “Path/folder”
• Cài thêm thư viện: !pip install <thư viện>
• Chạy file python: !python <file.py>
• Chú ý: Nếu chạy file python có tham số thì thực hiện giống như tại terminal của máy cá
nhân nhưng thêm dấu chấm thang (!) đầu lệnh
28
11. Triển khai Model Deep Learning trên Colab
29
• Một số model pre-trained dùng cho phân lớp ảnh được tích hợp trong Keras
• VGGNet
• Là mạng CNN sử dụng nhận dạng hình ảnh quy mô lớn
• Được giới thiệu bởi Simonyan và Zisserman vào năm 2014
• VGGNet có nhiều phiên bản
• VGGNet16  sử dụng 16 layer 
• VGGNet19  sử dụng 19 layer
• …
12. Sử dụng Model Deep Learning pre-trained trong Keras
30
12. Sử dụng Model Deep Learning pre-trained trong Keras
• VGGNet
31
• ResNet50
• Công bố vào năm 2015 bởi He và các cộng sự  Được cập nhật vào năm 2016
• Kiến trúc mạng có 50 lớp
• Sử dụng kỹ thuật Residual Block  Kết nối tắt giữa các lớp
12. Sử dụng Model Deep Learning pre-trained trong Keras
32
• ResNet50
12. Sử dụng Model Deep Learning pre-trained trong Keras
33
• Inception V3
• Được giới thiệu bởi Szegedy và các cộng sự vào năm 2014
• Sử dụng mô đun Inception
•  Giảm chiều của kênh
•  Rút trích đặc trưng multi-level
12. Sử dụng Model Deep Learning pre-trained trong Keras
34
• Inception V3
12. Sử dụng Model Deep Learning pre-trained trong Keras
35
• Xception
• Được giới thiệu bởi François Chollet vào năm 2016
• Xception là một phần mở rộng của kiến trúc Inception
• Sử dụng kỹ thuật depthwise separable convolutions  tách lớp tích chập 2D thành
tích chập 1D
12. Sử dụng Model Deep Learning pre-trained trong Keras
36
• Tập dữ liệu ImageNet
• Do Fei-Fei Li công bố vào năm 2009 Hội nghị về Thị giác Máy tính và Nhận dạng Mẫu
(CVPR) ở Florida
• ImageNet chứa hơn 20 nghìn danh mục
• Hơn 14 triệu các hình ảnh đã được gán nhãn
• Và được dùng để train các model được tích hợp trong Keras
• https://image-net.org/challenges/LSVRC/2014/index#data
12. Sử dụng Model Deep Learning pre-trained trong Keras
• Sử dụng các thư viện trong Keras
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
• Định nghĩa từ điển chứa các model
 MODELS = { "vgg16": VGG16, 
 "vgg19": VGG19, 
 "inception": InceptionV3, 
 "xception": Xception, 
 "resnet": ResNet50 31 } 
37
12. Sử dụng Model Deep Learning pre-trained trong Keras
• Sử dụng các thư viện trong Keras: Có 2 cách
• Cách 1:
• Lưu model về thành file *hdf5
• Sau đó Load model để dùng
• Cách 2: Load model sử dụng trực tiếp
• Load model về bộ nhớ và thực hiện dự đoán phân lớp ảnh (diễn ra cùng thời điểm)
• Thực hành sử dụng các model pre-trained đã tích hợp trong Keras
38
12. Sử dụng Model Deep Learning pre-trained trong Keras
from keras.applications.vgg19 import VGG19
model = VGG19()
model.save("modelvgg19.hdf5")
model.summary() # Hiển thị tóm tắt các 
tham số của model
