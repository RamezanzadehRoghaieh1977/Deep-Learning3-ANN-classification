# ------------------- کتابخانه‌ها -------------------
from keras.models import Sequential                     # برای ساخت مدل ترتیبی (Sequential)
from keras.layers import Dense, Flatten                 # لایه‌های Dense و Flatten
import tensorflow as tf                                 # کتابخانه TensorFlow
from matplotlib import pyplot as plt
%matplotlib widget                                      # فعال‌سازی حالت تعاملی matplotlib

# ------------------- بارگذاری دیتاست Fashion-MNIST -------------------
fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
train_images.shape                                      # ابعاد داده‌های آموزش
test_images.shape                                       # ابعاد داده‌های تست
test_images[0]                                          # نمایش ماتریس اولین تصویر
plt.close()
plt.imshow(train_images[1])                             # نمایش تصویر دوم دیتاست آموزش
plt.show()
print(train_labels[2])                                  # نمایش برچسب سومین تصویر

# ------------------- نام کلاس‌ها -------------------
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ------------------- نرمال‌سازی داده -------------------
train_images = train_images / 255.0                     # مقیاس‌گذاری به [0,1]
test_images = test_images / 255.0
# train_images, test_images = train_images / 255.0, test_images / 255.0   # نرمال‌سازی داده‌ها

# ------------------- تعریف مدل -------------------
model = Sequential()
# model.add(Dense(128, input_dim=784 , activation='relu'))   # روش جایگزین (کامنت شده)
# model.add(Dense(10))                                       # لایه خروجی ساده
model.add(Flatten(input_shape=(28,28)))                 # صاف کردن تصویر 28x28 به 784 ویژگی
model.add(Dense(128, activation='relu'))                # لایه مخفی با 128 نرون
model.add(Dense(10))                                    # لایه خروجی (10 کلاس)

# ------------------- کامپایل مدل -------------------
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
model.summary()                                         # نمایش معماری مدل

# ------------------- آموزش مدل -------------------
h = model.fit(train_images, train_labels, epochs=10)    # آموزش در 10 دوره (epoch)

# ------------------- پیش‌بینی روی داده تست -------------------
out1 = model.predict(test_images)                       # پیش‌بینی روی داده تست
out1[1]                                                 # نتایج پیش‌بینی برای نمونه دوم
class_names[test_labels[1]]                             # نام کلاس واقعی نمونه دوم
plt.close()
plt.imshow(test_images[1])                              # نمایش تصویر دوم داده تست
plt.show()

# ------------------- نمایش گرافیکی مدل -------------------
from keras_visualizer import visualizer
visualizer(model, file_name='model2', file_format='png')

# ------------------- تست مدل روی تصویر واقعی -------------------
import cv2
img = cv2.imread('boot2.png')                           # خواندن تصویر جدید
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # تبدیل به خاکستری
img.shape                                               # نمایش ابعاد تصویر
plt.close()
plt.imshow(img)                                         # نمایش تصویر
plt.show()

# ------------------- پیش‌پردازش تصویر -------------------
import numpy as np
img2 = np.array([img])                                  # تبدیل به آرایه
img2.shape                                              # نمایش ابعاد
img2 = img2 / 255                                       # نرمال‌سازی
out2 = model.predict(img2)                              # پیش‌بینی مدل روی تصویر
out2[0]                                                 # بردار خروجی (احتمالات هر کلاس)

# ------------------- پیدا کردن بیشترین احتمال -------------------
m = -1000
o2 = out2[0]
ind = -1
for i in range(len(o2)):                                # جستجوی بیشترین احتمال
    if o2[i] > m:
        m = o2[i]
        ind = i

print(class_names[ind])                                 # نمایش نام کلاس پیش‌بینی‌شده

# هم داده‌ی استاندارد Fashion-MNIST رو آموزش می‌ده.
# هم روی یک تصویر واقعی مثل boot2.png تست می‌کنه.