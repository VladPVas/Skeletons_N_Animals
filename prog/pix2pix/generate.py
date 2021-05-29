import tensorflow as tf
tf.keras.backend.clear_session()
import gc, os, sys
gc.collect()
abs_path = os.path.dirname(sys.executable)

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte

from pix2pix.model import crt_G, IMAGE_SIZE

g_model = crt_G()
g_model.trainable = False

g_last_weights = abs_path+'/pix2pix/g_last_weights.h5'
g_model.load_weights(g_last_weights, by_name = True, skip_mismatch = True)

def gen(image):
    inp_img_file = image
    inp_img = imread(inp_img_file)
    h,w,_ = inp_img.shape

    c_inp_img = img_as_ubyte(resize(inp_img,(IMAGE_SIZE,IMAGE_SIZE)))
    c_inp_img = ((c_inp_img[np.newaxis,...]/127.5) - 1.)
    pr_img = g_model.predict(c_inp_img)
    pr_img = np.uint8((pr_img+1.)*127.5)
    pr_img = img_as_ubyte(resize(pr_img[0],(h,w)))
    pr_img = resize(pr_img,(h,w))
    pr_img = img_as_ubyte(pr_img)
    return pr_img
