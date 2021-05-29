
import os 
from imgaug import augmenters as iaa 
import numpy as np 
from skimage.io import imread 
from skimage.transform import resize 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from skimage.color import gray2rgb, rgba2rgb, rgb2gray 
from skimage.util import img_as_ubyte 
from model import IMAGE_SIZE 


aug_img = iaa.Sequential([
    iaa.Sometimes(0.5,[ 
        iaa.ScaleX((0.75,1.25),mode='constant',fit_output=False,cval=255), 
        iaa.ScaleY((0.75,1.25),mode='constant',fit_output=False,cval=255), 
        iaa.ShearX((-15,15),mode='constant',fit_output=False,cval=255), 
        iaa.ShearY((-15,15),mode='constant',fit_output=False,cval=255), 
        iaa.Fliplr(0.5), 
        iaa.Flipud(0.5), 
        iaa.geometric.Rotate([0,90,180,270],mode='constant',fit_output=False,cval=255), 
        iaa.Crop(percent = (0.,0.10))
        ])
    ])


def get_num_batches(img_files, batch_size):
    return np.int32(len(img_files))//batch_size


def generate_batch(data_t, shuffle = True, aug = True, batch_size = 8):
    data_t = np.array(data_t)
    while True:
        if shuffle: data_t = np.random.permutation(data_t)
        num_batches = len(data_t)//batch_size
        for i in range(num_batches):
            data_batch = data_t[i:(i+batch_size)]
            input_X = []
            target_X = []
            for img_files in data_batch:
                input_img_file, output_img_file = img_files
                input_img = imread(input_img_file)
                target_img = imread(output_img_file) 
                if len(input_img.shape)==2: input_img = img_as_ubyte(gray2rgb(input_img))
                if input_img.shape[-1]==4: input_img = img_as_ubyte(rgba2rgb(input_img))
                if len(target_img.shape)==2: target_img = img_as_ubyte(gray2rgb(target_img))
                if target_img.shape[-1]==4: target_img = img_as_ubyte(rgba2rgb(target_img))

                input_img = img_as_ubyte(resize(input_img,(IMAGE_SIZE,IMAGE_SIZE)))
                target_img = img_as_ubyte(resize(target_img,(IMAGE_SIZE,IMAGE_SIZE)))
                input_X.append(input_img)
                target_X.append(target_img)

            if aug:
                aug_res = aug_img(images = np.concatenate([input_X,target_X],axis=3))
                input_X = aug_res[:,:,:,:3]
                target_X = aug_res[:,:,:,3:]

            input_X = np.clip(np.float32(input_X),0.,255.)
            target_X = np.clip(np.float32(target_X),0.,255.)
            input_X = (input_X/127.5) - 1.
            target_X = (target_X/127.5) - 1.
            yield input_X, target_X


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
