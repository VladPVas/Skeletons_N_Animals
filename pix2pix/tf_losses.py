import tensorflow as tf

base_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True, label_smoothing=0.15)


def d_loss(true_map, fake_map):
    real_loss = base_loss(tf.ones_like(true_map), true_map)
    fake_loss = base_loss(tf.zeros_like(fake_map), fake_map)
    total_loss = real_loss + fake_loss
    return total_loss


def g_loss(fake_map,fake_img,true_img):
    fake_loss = base_loss(tf.ones_like(fake_map), fake_map)
    l1_loss = tf.reduce_mean(tf.abs(true_img - fake_img))
    total_loss = fake_loss + (100.*l1_loss)
    return total_loss, l1_loss



