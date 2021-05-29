'''
Данный модуль включает в себя базовые функции по препроцессингу исходных данных:
    - загрузка изображения;
    - аугментация;
    - расчет откликов по сетке.

Реализация преимущественно на opencv, numpy - как наиболее быстрые варианты
'''

import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from imgaug import augmenters as iaa # для аугментации
import numpy as np # дл яработы с матрицами и векторами
from skimage.io import imread # быстрый загрузчик в формате RGB
from skimage.transform import resize # изменение размера изображения
import matplotlib.pyplot as plt # для графиков
from matplotlib.ticker import MaxNLocator
from skimage.color import gray2rgb, rgba2rgb, rgb2gray # для преобразования исходных данных изображений
from skimage.util import img_as_ubyte # для преобразоваия типа данных изображений
from model import IMAGE_SIZE # целевое разрешение изображений

# Аугментатор входа
aug_img = iaa.Sequential([
    iaa.Sometimes(0.5,[ # срабатывает в 50% случаев
        iaa.ScaleX((0.75,1.25),mode='constant',fit_output=False,cval=255), # случайное масштабирование по Х
        iaa.ScaleY((0.75,1.25),mode='constant',fit_output=False,cval=255), # случайное масштабирование по Y
        iaa.ShearX((-15,15),mode='constant',fit_output=False,cval=255), # случайное смещение по X
        iaa.ShearY((-15,15),mode='constant',fit_output=False,cval=255), # случайное смещение по Y
        iaa.Fliplr(0.5), # зеркалирование 
        iaa.Flipud(0.5), # зеркалирование
        iaa.geometric.Rotate([0,90,180,270],mode='constant',fit_output=False,cval=255), # случайный поворот на углы кратные 90 градусов
        iaa.Crop(percent = (0.,0.10))
        ])#,
    #iaa.Sometimes(0.5,[ # срабатывает в 50% случаев
        #iaa.SomeOf(1,[ # выбирается только 1 метод
            #iaa.Dropout((0.001, 0.01), per_channel=0.5), # случайное "выбивание" отдельных пикселей
            #iaa.Multiply((0.95, 1.05), per_channel = False), # случайное изменение яркости
            #iaa.LinearContrast((0.95, 1.05), per_channel = False), # случайное изменение контрастности
            #iaa.GaussianBlur(sigma=(0,3.0)) # Гаусово размытие
            #])
        #])
    ])

# Только для отладки
'''
import cv2
img = cv2.imread('./dataset/skeletons/1.jpg')
img = np.concatenate([img,img],axis=2)

img = aug_img(image=img)
plt.imshow(img[:,:,:3])
plt.show()
plt.imshow(img[:,:,3:])
plt.show()
'''

# Получить число шагов для выборки
def get_num_batches(img_files, batch_size):
    return np.int32(len(img_files))//batch_size


# функция для генерации батча в процессе обучения по списку файлов
# Указывается список файлов, нужно ли его перемешивать, нужна ли аугментация
# Выход  - матрица изображений X
def generate_batch(data_t, shuffle = True, aug = True, batch_size = 8):
    data_t = np.array(data_t)
    while True:
        # полное перемешивание
        if shuffle: data_t = np.random.permutation(data_t)
        # расчет числа батчей
        num_batches = len(data_t)//batch_size
        # для каждого батча выполнить:
        for i in range(num_batches):
            # загрузить батч пар имен файлов
            data_batch = data_t[i:(i+batch_size)]
            # загрузить файлы ввиде списков
            input_X = []
            target_X = []
            for img_files in data_batch:
                input_img_file, output_img_file = img_files
                # загрузка изображений
                input_img = imread(input_img_file)
                target_img = imread(output_img_file) 
                # проверка числа каналов и корректировка если необходимо
                if len(input_img.shape)==2: input_img = img_as_ubyte(gray2rgb(input_img))
                if input_img.shape[-1]==4: input_img = img_as_ubyte(rgba2rgb(input_img))
                if len(target_img.shape)==2: target_img = img_as_ubyte(gray2rgb(target_img))
                if target_img.shape[-1]==4: target_img = img_as_ubyte(rgba2rgb(target_img))
                # устанавливаем целевое разрешение 512х512
                input_img = img_as_ubyte(resize(input_img,(IMAGE_SIZE,IMAGE_SIZE)))
                target_img = img_as_ubyte(resize(target_img,(IMAGE_SIZE,IMAGE_SIZE)))
                input_X.append(input_img)
                target_X.append(target_img)
            # аугментация входа и выхода если задана
            if aug:
                # слияние по каналам и аугментация
                aug_res = aug_img(images = np.concatenate([input_X,target_X],axis=3))
                # поканальное разделение на вход и выход
                input_X = aug_res[:,:,:,:3]
                target_X = aug_res[:,:,:,3:]
            # черно-белое кодирование
            #input_X = [img_as_ubyte(rgb2gray(img)) for img in input_X]
            #target_X = [img_as_ubyte(rgb2gray(img)) for img in target_X]
            # батчи для входа и выхода генератора
            input_X = np.clip(np.float32(input_X),0.,255.)
            target_X = np.clip(np.float32(target_X),0.,255.)
            input_X = (input_X/127.5) - 1.
            target_X = (target_X/127.5) - 1.
            # выход генератора
            yield input_X, target_X


# Только для отладки
'''
import glob
# получить список всех файлов изображений для входа
train_t = np.loadtxt('./dataset/train.txt', dtype=np.object,delimiter="\t")[:1]
train_t = np.repeat(train_t,5,axis=0)

g = generate_batch(train_t,shuffle = False, aug = True, batch_size=1)
num_train_steps = 5#np.uint32(len(train_t))
#st = time.time()
for i in range(num_train_steps):
    X,Y = next(g)
    plt.imshow((X[0,...,0]+1)*127, cmap='gray')
    plt.show()
    plt.imshow((Y[0,...,0]+1)*127, cmap='gray')
    plt.show()
'''

# Визуализация тренировки и валидации - W
def viz_train_val(train_g_loss,train_d_loss,
                  train_l1_loss,
                  out_file = './train_val_learn.png'):
    fig, axs = plt.subplots(3,constrained_layout=True)
    fig.suptitle('PIP2PIX GAN losses')
    axs[0].plot(train_g_loss, color="r")
    axs[0].set_title('training g_loss')
    axs[0].set(xlabel='epoches', ylabel='loss')
    axs[0].grid()
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    axs[1].plot(train_d_loss, color="b")
    axs[1].set_title('training d_loss')
    axs[1].set(xlabel='epoches', ylabel='loss')
    axs[1].grid()
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    axs[2].plot(train_l1_loss, color="r")
    axs[2].set_title('training l1_loss')
    axs[2].set(xlabel='epoches', ylabel='loss')
    axs[2].grid()
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.savefig(out_file, dpi=300)
    plt.show()
    return None

#viz_train_val([0.1,1,2., 2,3],[2,2,4],[4,5,6])



