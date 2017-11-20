from keras.callbacks import TensorBoard, Callback

class TensorBoardGen(TensorBoard):
    '''
        Updated TensorBoard callback to work with validation generator
        Exmaple of use:
            TensorBoardGen('./logs', histogram_freq=1, write_images=True, validation_generator=generator(summary_batch))
    '''

    def __init__(self, *args, **kwargs):
        if 'validation_generator' in kwargs:
            self.validation_generator = kwargs['validation_generator']
            del kwargs['validation_generator']
        super(TensorBoardGen, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        res = self.validation_generator.next()
        self.validation_data = [
            res[0], res[1], [1.] * len(res[0])
        ]
        if self.model.uses_learning_phase:
            self.validation_data += [0.]

        super(TensorBoardGen, self).on_epoch_end(epoch, logs)
        
        
class TFCheckpointCallback(Callback):

    def __init__(self, ch_dir='./ch'):
        self.saver = tf.train.Saver()
        self.sess = K.get_session()
        self.ch_dir = ch_dir

    def on_epoch_end(self, epoch, logs=None):
        self.saver.save(self.sess, os.path.join(self.ch_dir, 'checkpoint.ckpt'), global_step=epoch)

    def on_train_begin(self, logs=None):
        if os.listdir(self.ch_dir) and 'checkpoint' in os.listdir(self.ch_dir)[0]:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.ch_dir))
            print('load weights: OK.')
