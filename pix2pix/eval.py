import tensorflow as tf
tf.keras.backend.clear_session()
import gc, os
gc.collect()

import numpy as np
from skimage.io import imread, imsave 
from skimage.transform import resize 
from skimage.util import img_as_ubyte 
import matplotlib.pyplot as plt 

from model import crt_G, IMAGE_SIZE


g_model = crt_G()
g_model.trainable = False
g_last_weights = './g_last_weights.h5'
if os.path.exists(g_last_weights):
    g_model.load_weights(g_last_weights, by_name = True, skip_mismatch = True)


inp_img_file = './test/90_inp.jpg'
inp_img = imread(inp_img_file)
h,w,_ = inp_img.shape 
plt.imshow(inp_img)
plt.show()

c_inp_img = img_as_ubyte(resize(inp_img,(IMAGE_SIZE,IMAGE_SIZE)))
c_inp_img = ((c_inp_img[np.newaxis,...]/127.5) - 1.) 
pr_img = g_model.predict(c_inp_img) 
pr_img = np.uint8((pr_img+1.)*127.5) 
pr_img = img_as_ubyte(resize(pr_img[0],(h,w))) 
plt.imshow(pr_img) 
plt.show()

out_img_file = './test/90_out.jpg'
out_img = imread(out_img_file)
out_img = img_as_ubyte(resize(out_img,(h,w)))
plt.imshow(out_img) # выводим на экран
plt.show()

pr_img = resize(pr_img,(h,w))
imsave(fname='./test/pred_'+os.path.basename(out_img_file),arr=img_as_ubyte(pr_img))
