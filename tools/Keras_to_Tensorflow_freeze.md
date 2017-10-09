# How to freeze you Keras model in Tensorflow model

First you need to create TF checkpoints and save you graph. Here is exameple how to do this:

```python
import keras
import keras.backend as K
import tensorflow as tf

class TFCkptCallback(keras.callbacks.Callback):

    def __init__(self, saver, sess):
        self.saver = saver
        self.sess = sess

    def on_epoch_end(self, epoch, logs=None):
        self.saver.save(self.sess, 'path_to_checkpoints/checkpoint.ckpt', global_step=epoch)


tf_graph = sess.graph
tf_saver = tf.train.Saver()
# saving weights
tf_ckpt_cb = TFCkptCallback(tf_saver, sess)

model.fit_generator(
    generator=traingen, 
    validation_data=testgen,
    steps_per_epoch=500,
    validation_steps=50,
    epochs=1,
    verbose=1,
    callbacks=[tf_ckpt_cb]
)

# saving graph
tf.train.write_graph(tf_graph.as_graph_def(), 'dir_path', 'graph.pb', as_text=False)

```


If you need to list all your tensors in the model, use this:
```python
[print(n.name) for n in sess.graph.as_graph_def().node]
```


To freeze model and create one file use this.
In Keras if you have Activation name sigmoid, and activation function is Sigmoid, the tensor name will be "tesnor_name/Function_name", for example sigmoid/Sigmoid.
```python
from tensorflow.python.tools import freeze_graph

MODEL_NAME = 'har'

input_graph_path = './weights/graph.pb'
checkpoint_path = './weights/checkpoint.ckpt-0'
output_frozen_graph_name = 'frozen.pb'
last_tensor_name = "sigmoid/Sigmoid"

freeze_graph.freeze_graph(input_graph_path, input_saver="",
                          input_binary=True, input_checkpoint=checkpoint_path, 
                          output_node_names=last_tensor_name, restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0", 
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")
```

Now you can import you model in TF runner or to mobile device.
