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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Параметры командной строки
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Запуск с параметрами:")  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # число эпох обучения
    parser.add_argument('--num_epochs', type=int, default=50)
    # файл для сохранения весов модели генератора на последней итерации обучения
    parser.add_argument('--g_last_weights', type=str, default='./g_last_weights.h5')
    # файл для сохранения весов модели дискриминатора на последней итерации обучения
    parser.add_argument('--d_last_weights', type=str, default='./d_last_weights.h5')
    # файл визуализации тренировки
    parser.add_argument('--viz_file', type=str, default='./train_val_learn.png')
    # начальный коэффициент обучения
    parser.add_argument('--g_lr', type=float, default=1e-4)
    # финальный итоговый коэффициент обучения
    parser.add_argument('--d_lr', type=float, default=1e-5)   
    # папка с исходными данными
    parser.add_argument('--batch_size', type=str, default=1)  
    # папка с исходными данными
    parser.add_argument('--source_folder', type=str, default='./dataset')
    # парсинг параметров        
    opt = parser.parse_args()

print(opt)
print("BATCH SIZE: {}; IMAGE SIZE {}".format(opt.batch_size, IMAGE_SIZE))
print("\n")
time.sleep(0.1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Основной процесс тренировки
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# получить список всех пар файлов для обучения и валидации
train_t = np.loadtxt(opt.source_folder+'/train.txt',delimiter='\t',dtype=object, encoding='utf8')#[:100]

# Корректировка числа файлов, исходя из размера батча
train_m = opt.batch_size-len(train_t)%opt.batch_size # число файлов, которые нужно добавить
# Все файлы будут присутствовать в списке + несколько дублированных
if train_m<opt.batch_size: train_t = np.concatenate([train_t,np.random.permutation(train_t)[:train_m]])

# При малых объёмах данных
#train_t = np.repeat(train_t,10,axis=0)
#val_t = np.repeat(val_t,10,axis=0)
print("Всего исходных сетов: {}".format(len(train_t)))

# число файлов для тернировки и валидации
train_n = len(train_t)
print("Число файлов в тренировочном наборе: {}".format(train_n))

# Определение генераторов данных для тестирования и валидации
train_gen = du.generate_batch(data_t = train_t, aug = True, shuffle = True)
# расчет числа батчей
num_train_steps = du.get_num_batches(train_t, opt.batch_size)

# Наилучший лосс/метрика по валидации
best_loss = np.Inf
best_mac = -np.Inf

# создаём модели генератора и дискриминатора
g_model = crt_G()
d_model = crt_D()

# пытаемся загрузить в них веса
if os.path.exists(opt.g_last_weights):
    g_model.load_weights(opt.g_last_weights, by_name = True, skip_mismatch = True) # загружаем все, что получится
if os.path.exists(opt.d_last_weights):
    d_model.load_weights(opt.d_last_weights, by_name = True, skip_mismatch = True) # загружаем все, что получится

# определяем оптимизаторы
g_optimizer = tf.keras.optimizers.Adam(opt.g_lr, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(opt.d_lr, beta_1=0.5)

# Тренировка на батче. Декоратор @tf.function нужен для компиляции
@tf.function
def train_step(input_X, target_X, epoch_num):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        # генерируем нужные нам изображения из исходных
        g_target_X = g_model(input_X, training=True)
        # получаем выходы дискриминатора
        true_map = d_model([input_X,target_X], training=True)
        fake_map = d_model([input_X,g_target_X], training=True)
        # считаем лоссы
        c_g_loss, l1_loss = g_loss(fake_map,g_target_X,target_X) # здесь желательно, чтобы были значения ближе к 1 (к истиным)
        c_d_loss = d_loss(true_map, fake_map) # здесь желательно, чтобы отклики отличались как можно сильнее
        # считаем градиенты
        grad_g = g_tape.gradient(c_g_loss, g_model.trainable_variables)
        grad_d = d_tape.gradient(c_d_loss, d_model.trainable_variables)
        # обновляем веса моделей
        g_optimizer.apply_gradients(zip(grad_g, g_model.trainable_variables))
        # обновление весов дискриминатора осуществляем реже
        #if (epoch_num+1)%3==0:
        d_optimizer.apply_gradients(zip(grad_d, d_model.trainable_variables))
    return c_g_loss, c_d_loss, l1_loss


# Массив лоссов и метрик после каждого цикла тренировки и валидации
train_g_loss = []
train_d_loss = []
train_l1_loss = []

# цикл тренировки и валидации
for epoch_num in range(opt.num_epochs):
    print('Epoch {}/{}'.format(epoch_num + 1,opt.num_epochs))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Тренировка
    progbar = tf.keras.utils.Progbar(target = num_train_steps, interval=0.5, 
                                     stateful_metrics= ['g_loss','d_loss','l1_loss'])
    # сумма значений по лоссам
    s_g_loss, s_d_loss, s_l1_loss = 0., 0., 0.
    # средние значения по лоссам
    mean_g_loss, mean_d_loss, mean_l1_loss = 0., 0., 0.
    for iter_num, (input_X,target_X) in enumerate(train_gen): 
        if iter_num == num_train_steps: break
        # получение лоссов
        c_g_loss, c_d_loss, c_l1_loss = train_step(input_X,target_X,epoch_num)
        # суммирование лоссов
        s_g_loss+=c_g_loss.numpy()
        s_d_loss+=c_d_loss.numpy()
        s_l1_loss+=c_l1_loss.numpy()
        # расчет средних лоссов
        mean_g_loss = s_g_loss/(iter_num+1)
        mean_d_loss = s_d_loss/(iter_num+1)
        mean_l1_loss = s_l1_loss/(iter_num+1)
        # прогресс
        progbar.update(iter_num+1,[('g_loss', mean_g_loss),
                                   ('d_loss', mean_d_loss),
                                   ('l1_loss', mean_l1_loss)])
    train_g_loss.append(mean_g_loss)
    train_d_loss.append(mean_d_loss)
    train_l1_loss.append(mean_l1_loss)
    g_model.save_weights(opt.g_last_weights)
    d_model.save_weights(opt.d_last_weights)
    

# Визуализация
du.viz_train_val(train_g_loss,train_d_loss, train_l1_loss)
print("Нейронная сеть обучена!")

'''
# проверка
import matplotlib.pyplot as plt # для графиков
X,Y = next(train_gen)
res = g_model.predict(X)
for i in range(len(res)):
    plt.imshow((X[i,:,:,0]+1)*127,cmap='gray')
    plt.show()
    plt.imshow((res[i,:,:,0]+1)*127,cmap='gray')
    plt.show()
    plt.imshow((Y[i,:,:,0]+1)*127,cmap='gray')
    plt.show()
'''      
