import tensorflow as tf
tf.keras.backend.clear_session()
import gc, os
gc.collect()

import numpy as np
from skimage.io import imread, imsave # быстрый загрузчик в формате RGB
from skimage.transform import resize # изменение размера изображения
from skimage.util import img_as_ubyte # для преобразоваия типа данных изображений
import matplotlib.pyplot as plt # для графиков

from model import crt_G, IMAGE_SIZE

# инициализируем модель генератора
g_model = crt_G()
g_model.trainable = False
# пытаемся загрузить в неё веса
g_last_weights = './g_last_weights.h5'
if os.path.exists(g_last_weights):
    g_model.load_weights(g_last_weights, by_name = True, skip_mismatch = True) # загружаем все, что получится

# загружаем изображение на вход генератора
inp_img_file = './test/90_inp.jpg'
inp_img = imread(inp_img_file)
h,w,_ = inp_img.shape # сохраняем пропорции
plt.imshow(inp_img) # выводим на экран
plt.show()
# генерируем изображение на основе входной картинки
c_inp_img = img_as_ubyte(resize(inp_img,(IMAGE_SIZE,IMAGE_SIZE))) # изменяем размер до "рабочего"
c_inp_img = ((c_inp_img[np.newaxis,...]/127.5) - 1.) # добавляем размерность и масштабируем
pr_img = g_model.predict(c_inp_img) # пропускаем через сеть
pr_img = np.uint8((pr_img+1.)*127.5) # демасштабирование и изменение типа данных
pr_img = img_as_ubyte(resize(pr_img[0],(h,w))) # изменяем размер до исходного и убираем лишнюю размерность
plt.imshow(pr_img) # выводим на экран
plt.show()
# загружаем контрольное изображение
out_img_file = './test/90_out.jpg'
out_img = imread(out_img_file)
out_img = img_as_ubyte(resize(out_img,(h,w))) # изменяем размер "до входного"
plt.imshow(out_img) # выводим на экран
plt.show()
# сохраняем сгенерированное изображение
pr_img = resize(pr_img,(h,w))
imsave(fname='./test/pred_'+os.path.basename(out_img_file),arr=img_as_ubyte(pr_img))
