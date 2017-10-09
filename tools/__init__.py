import keras
import keras.backend as K
import tensorflow as tf


class TFCheckpointCallback(keras.callbacks.Callback):

    def __init__(self, saver, sess, path):
        self.saver = saver
        self.sess = sess
        self.path

    def on_epoch_end(self, epoch, logs=None):
        self.saver.save(self.sess, self.path, global_step=epoch)
