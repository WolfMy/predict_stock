import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class LSTM():
    def __init__(self, name_scope, input_size, time_steps, batch_size=16, hidden_units = 128, learning_rate=0.01, epoches=100):
        self.name_scope = name_scope
        self.input_size = input_size
        self.time_steps = time_steps
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoches = epoches
        self.NetInit()
    
    def weight_variable(self, shape):
        with tf.name_scope('weights'):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')
    
    def bias_variable(self, shape):
        with tf.name_scope('biases'):
            return tf.Variable(tf.constant(0.1, shape=shape))

    def NetInit(self):
        with tf.name_scope('Inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.time_steps, self.input_size], name='x_input')
            self.y_ = tf.placeholder(tf.float32, [None, self.input_size], name='y_input')

        weights = {
            'in': self.weight_variable([self.input_size, self.hidden_units]),
            'out': self.weight_variable([self.hidden_units, self.input_size])
        }
        biases = {
            'in': self.bias_variable([self.hidden_units, ]),
            'out': self.bias_variable([self.input_size, ])
        }
        
        # RNN
        x = tf.reshape(self.x, [-1, self.input_size])
        with tf.name_scope('Wx_plus_b'):
            x_in = tf.matmul(x, weights['in']) + biases['in']
        x_in = tf.reshape(x_in, [-1, self.time_steps, self.hidden_units])

        
        # basic LSTM Cell.
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units, forget_bias=1.0)
        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, x_in, initial_state=init_state, time_major=False, scope=self.name_scope)

        self.prediction = tf.matmul(final_state[1], weights['out']) + biases['out']    # shape = (batch_size, input_size)

        with tf.name_scope('loss'):
            self.loss = tf.losses.mean_squared_error(labels=self.y_, predictions=self.prediction)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.merg_op = tf.summary.merge_all()

    def handle_predict_data(self, predict_data):
        predict_data = predict_data[np.newaxis, :]
        return_data = predict_data
        for i in range(self.batch_size-1):
            return_data = np.vstack([return_data, predict_data])
        return return_data  # (batch_size, time_steps, input_size)

    def get_batches(self, dataset):
        length = dataset.shape[0]
        x_batches = []
        y_batches = []
        for i in range(length-self.batch_size-self.time_steps-1):
            batch = []
            batch_label = []
            for j in range(self.batch_size):
                batch.append(dataset[i+j:i+j+self.time_steps])
                batch_label.append(dataset[i+j+self.time_steps])
            x_batches.append(batch)
            y_batches.append(batch_label)
        x_batches = np.array(x_batches)
        y_batches = np.array(y_batches)
        return x_batches, y_batches

    def train(self, dataset):
        for i in range(self.epoches+1):
            x_batches, y_batches = self.get_batches(dataset)
            assert len(x_batches) == len(y_batches)
            for j in range(len(x_batches)):
                train_data = x_batches[j]
                train_label = y_batches[j]
                self.sess.run (self.train_op, feed_dict = {self.x: train_data, self.y_: train_label})
            if i%50 == 0:
                loss = self.get_loss(x_batches[0], y_batches[0])
                print("epoch %d, loss:%f" % (i, loss))
        print("训练完成！")

    def get_loss(self, data, label):
        return self.sess.run(self.loss, feed_dict={self.x: data, self.y_: label})
    
    def predict(self, data):
        data = self.handle_predict_data(data)
        return self.sess.run(self.prediction, feed_dict={self.x: data})[-1].reshape(1, -1)

if __name__ == "__main__":
    dataset = pd.read_csv(filepath_or_buffer=r'/Users/mouyu/Desktop/600303.SH_08_18.csv')
    dataset = dataset.values
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    #print(dataset)

    train_dataset = dataset[:40]
    test_dataset = dataset[-13:-1]
    # train
    lstm = LSTM(name_scope='rnn', input_size=dataset.shape[1], time_steps=12, epoches=100)
    lstm.train(dataset=train_dataset)

    # test
    print("prediction on test_dataset:")
    print(scaler.inverse_transform(lstm.predict(test_dataset)))
    print("y_test:")
    print(scaler.inverse_transform(dataset[-1].reshape(1, -1)))

