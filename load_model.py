import tensorflow as tf
import numpy as np


class Model_Mnist:
    def __init__(self):
        self.graph_mnist = tf.Graph()

    def load_model(self, meta_file, folder_name):
        with self.graph_mnist.as_default():
            self.sess_mnist = tf.Session(graph=self.graph_mnist)
            saver = tf.compat.v1.train.import_meta_graph(meta_file)
            saver.restore(self.sess_mnist,
                          tf.train.latest_checkpoint(folder_name))
            graph_1 = tf.compat.v1.get_default_graph()

            self.weights = {
                'w1': graph_1.get_tensor_by_name("w1:0"),
                'w2': graph_1.get_tensor_by_name("w2:0"),
                'w3': graph_1.get_tensor_by_name("w3:0"),
                'w4': graph_1.get_tensor_by_name("w4:0"),
                
            }
            self.biases = {
                'b1': graph_1.get_tensor_by_name("b1:0"),
                'b2': graph_1.get_tensor_by_name("b2:0"),
                'b3': graph_1.get_tensor_by_name("b3:0"),
                'b4': graph_1.get_tensor_by_name("b4:0"),
                
            }

    def net(self, x, weights, biases):
        dense_1 = tf.matmul(x, weights['w1'])+biases['b1']
        relu_1 = tf.nn.relu(dense_1)
        dense_2 = tf.matmul(relu_1, weights['w2'])+biases['b2']
        relu_2 = tf.nn.relu(dense_2)
        dense_3 = tf.matmul(relu_2, weights['w3'])+biases['b3']
        relu_3 = tf.nn.relu(dense_3)
        dense_4 = tf.matmul(relu_3, weights['w4'])+biases['b4']
        return dense_4

    def predict(self, x):
        with self.sess_mnist.as_default():
            with self.graph_mnist.as_default():
                X = tf.placeholder(tf.float32, shape=(None, 784))
                pred = self.net(X, self.weights, self.biases)
                result = self.sess_mnist.run(pred, feed_dict={X: x})
                return np.argmax(result[0])


class Model_Written:
    def __init__(self):

        self.graph_written = tf.Graph()

    def load_model(self, meta_file, folder_name):
        with self.graph_written.as_default():
            self.sess_written = tf.Session(graph=self.graph_written)
            saver = tf.compat.v1.train.import_meta_graph(meta_file)
            saver.restore(self.sess_written,
                          tf.train.latest_checkpoint(folder_name))
            graph_2 = tf.compat.v1.get_default_graph()

            self.weights = {
                'w1': graph_2.get_tensor_by_name("W1:0"),
                'w2': graph_2.get_tensor_by_name("W2:0"),
                'w3': graph_2.get_tensor_by_name("W3:0"),
                'w4': graph_2.get_tensor_by_name("W4:0"),
                'w5': graph_2.get_tensor_by_name("W5:0"),
            }
            self.biases = {
                'b1': graph_2.get_tensor_by_name("B1:0"),
                'b2': graph_2.get_tensor_by_name("B2:0"),
                'b3': graph_2.get_tensor_by_name("B3:0"),
                'b4': graph_2.get_tensor_by_name("B4:0"),
                'b5': graph_2.get_tensor_by_name("B5:0"),
            }

    def net(self, x, weights, biases):
        dense_1 = tf.matmul(x, weights['w1']) + biases['b1']
        relu_1 = tf.nn.relu(dense_1)
        dense_2 = tf.matmul(relu_1, weights['w2'])+biases['b2']
        relu_2 = tf.nn.relu(dense_2)
        dense_3 = tf.matmul(relu_2, weights['w3'])+biases['b3']
        relu_3 = tf.nn.relu(dense_3)
        dense_4 = tf.matmul(relu_3, weights['w4'])+biases['b4']
        relu_4 = tf.nn.relu(dense_4)
        dense_5 = tf.matmul(relu_4, weights['w5'])+biases['b5']
        return tf.nn.softmax(dense_5)

    def predict(self, x):
        with self.sess_written.as_default():
            with self.graph_written.as_default():
                X = tf.placeholder(tf.float32, shape=(None, 784))
                pred = self.net(X, self.weights, self.biases)
                result = self.sess_written.run(pred, feed_dict={X: x})
                return np.argmax(result[0])
