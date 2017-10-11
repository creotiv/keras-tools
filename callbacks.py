from keras.callbacks import TensorBoard

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
