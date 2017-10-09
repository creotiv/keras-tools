import keras
import keras.backend as K
import tensorflow as tf

def weighted_loss(y_true, y_pred):
    mask = K.expand_dims(y_true[:, :, :, 0])
    edge = K.expand_dims(y_true[:, :, :, 1])

    sq = K.square(y_pred - mask)
    mse = K.mean(sq, axis=-1)
    mse2 = K.mean(sq * edge, axis=-1)
    return mse + mse2

