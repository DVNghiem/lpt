from load_model import Model_Mnist, Model_Written
import tensorflow as tf
mnist = Model_Mnist()
written = Model_Written()
mnist.load_model('model_mnist/model.ckpt.meta', 'model_mnist')
written.load_model('model_hand_written/model.ckpt.meta', 'model_hand_written')

t=tf.Graph()
with t.as_default():
    sess = tf.Session(graph = t)
    


