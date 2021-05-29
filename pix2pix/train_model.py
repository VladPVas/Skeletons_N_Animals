import tensorflow as tf
tf.keras.backend.clear_session()
import gc
gc.collect()

import argparse, time, os
import numpy as np

import data_utils as du
from model import IMAGE_SIZE
from model import crt_G, crt_D
from tf_losses import d_loss, g_loss


print("Запуск с параметрами:")  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--g_last_weights', type=str, default='./g_last_weights.h5')
    parser.add_argument('--d_last_weights', type=str, default='./d_last_weights.h5')
    parser.add_argument('--viz_file', type=str, default='./train_val_learn.png')
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=5e-5)   
    parser.add_argument('--batch_size', type=str, default=1)  
    parser.add_argument('--source_folder', type=str, default='./dataset')
   
    opt = parser.parse_args()

print(opt)
print("BATCH SIZE: {}; IMAGE SIZE {}".format(opt.batch_size, IMAGE_SIZE))
print("\n")
time.sleep(0.1)


train_t = np.loadtxt(opt.source_folder+'/train.txt',delimiter='\t',dtype=object, encoding='utf8')#[:100]

train_m = opt.batch_size-len(train_t)%opt.batch_size
if train_m<opt.batch_size: train_t = np.concatenate([train_t,np.random.permutation(train_t)[:train_m]])

#train_t = np.repeat(train_t,10,axis=0)
#val_t = np.repeat(val_t,10,axis=0)
print("Всего исходных сетов: {}".format(len(train_t)))

train_n = len(train_t)
print("Число файлов в тренировочном наборе: {}".format(train_n))

train_gen = du.generate_batch(data_t = train_t, aug = True, shuffle = True)
num_train_steps = du.get_num_batches(train_t, opt.batch_size)

best_loss = np.Inf
best_mac = -np.Inf

g_model = crt_G()
d_model = crt_D()

if os.path.exists(opt.g_last_weights):
    g_model.load_weights(opt.g_last_weights, by_name = True, skip_mismatch = True) # загружаем все, что получится
if os.path.exists(opt.d_last_weights):
    d_model.load_weights(opt.d_last_weights, by_name = True, skip_mismatch = True) # загружаем все, что получится

g_optimizer = tf.keras.optimizers.Adam(opt.g_lr, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(opt.d_lr, beta_1=0.5)

@tf.function #для компиляции
def train_step(input_X, target_X, epoch_num):
    with tf.GradientTape(persistent=True) as tape:
        g_target_X = g_model(input_X, training=True)
        true_map = d_model([input_X,target_X], training=True)
        fake_map = d_model([input_X,g_target_X], training=True)
        
        c_g_loss, l1_loss = g_loss(fake_map,g_target_X,target_X)
        c_d_loss = d_loss(true_map, fake_map)
        grad_g = tape.gradient(c_g_loss, g_model.trainable_variables)
        grad_d = tape.gradient(c_d_loss, d_model.trainable_variables)
        g_optimizer.apply_gradients(zip(grad_g, g_model.trainable_variables))
        d_optimizer.apply_gradients(zip(grad_d, d_model.trainable_variables))
    return c_g_loss, c_d_loss, l1_loss

train_g_loss = []
train_d_loss = []
train_l1_loss = []


for epoch_num in range(opt.num_epochs):
    print('Epoch {}/{}'.format(epoch_num + 1,opt.num_epochs))


    progbar = tf.keras.utils.Progbar(target = num_train_steps, interval=0.5, 
                                     stateful_metrics= ['g_loss','d_loss','l1_loss'])

    s_g_loss, s_d_loss, s_l1_loss = 0., 0., 0.
    mean_g_loss, mean_d_loss, mean_l1_loss = 0., 0., 0.
    for iter_num, (input_X,target_X) in enumerate(train_gen): 
        if iter_num == num_train_steps: break
        c_g_loss, c_d_loss, c_l1_loss = train_step(input_X,target_X,epoch_num)
        s_g_loss+=c_g_loss.numpy()
        s_d_loss+=c_d_loss.numpy()
        s_l1_loss+=c_l1_loss.numpy()
        mean_g_loss = s_g_loss/(iter_num+1)
        mean_d_loss = s_d_loss/(iter_num+1)
        mean_l1_loss = s_l1_loss/(iter_num+1)

        progbar.update(iter_num+1,[('g_loss', mean_g_loss),
                                   ('d_loss', mean_d_loss),
                                   ('l1_loss', mean_l1_loss)])
    train_g_loss.append(mean_g_loss)
    train_d_loss.append(mean_d_loss)
    train_l1_loss.append(mean_l1_loss)
    g_model.save_weights(opt.g_last_weights)
    d_model.save_weights(opt.d_last_weights)
    

du.viz_train_val(train_g_loss,train_d_loss, train_l1_loss)
print("Нейронная сеть обучена!")
