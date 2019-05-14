import tensorflow as tf
import pandas as pd
import numpy as np
from MACD_RSI import init_train_data

class BpNet():
    def __init__(self, input_size, num_classes, learning_rate=0.001, batch_size=32, num_epochs=1000):
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.NetInit()
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        Weights = self.weight_variable([in_size, out_size])
        biases = self.bias_variable([out_size])
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def NetInit(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        self.layer_1 = self.add_layer(self.x, self.input_size, 16, activation_function=tf.nn.relu)
        self.layer_2 = self.add_layer(self.layer_1, 16, 8, activation_function=tf.nn.relu)
        self.y = self.add_layer(self.layer_2, 8, self.num_classes, activation_function=tf.nn.softmax)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_, logits = self.y))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.sess = tf.Session()

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

    def train(self, train_data, train_label):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        epoch = 0
        try:
            while not coord.should_stop():
                # 获取训练用的每一个batch中batch_size个样本和标签
                batch_input, batch_label = self.sess.run([train_data, train_label])
                self.sess.run (self.train_step, feed_dict = {self.x: batch_input, self.y_: batch_label})
                epoch = epoch + 1
        except tf.errors.OutOfRangeError:  # num_epochs 次数用完会抛出此异常
            print(self.sess.run(self.accuracy, feed_dict={self.x: batch_input, self.y_: batch_label}))
        finally:
            # 协调器coord发出所有线程终止信号
            coord.request_stop()
            coord.join(threads)  # 把开启的线程加入主线程，等待threads结束

    def get_accuracy(self, data, label):
        return self.sess.run(self.accuracy, feed_dict={self.x: data, self.y_: label})
    
    def predict(self, data):
        return self.sess.run(self.y, feed_dict={self.x: data})

def get_batch(data, label, batch_size, num_epochs):
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=num_epochs, shuffle=True, capacity=32)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
    return x_batch, y_batch

if __name__=='__main__':
    start_date = '2018-11-20'
    end_date = '2019-03-01'
    stock_list = ['603000', '300492', '600528', '002230', '601688']
    df = pd.DataFrame(stock_list, columns=['stock'])
    train_acc = []
    model = []
    for stock in stock_list:
        data, label = init_train_data(stock, start_date, end_date)
        x_batch, y_batch = get_batch(data, label, batch_size=32,num_epochs=5000)
        bpnet = BpNet(input_size=data.shape[1], num_classes=3, learning_rate=0.001, batch_size=32, num_epochs=5000)
        bpnet.train(x_batch, y_batch)
        train_acc.append(bpnet.get_accuracy(data, label))
        model.append(bpnet)
    df['train_acc'] = train_acc
    df['model'] = model
    print(df)
    df = df.sort_values(['train_acc'], ascending=False).reset_index(drop=True)[:3]
    print(df)

    test_data, test_label = init_train_data('603000','2019-01-01','2019-03-03')
    print("truth:", np.argmax(test_label,1))
    print("prediction:", np.argmax(df.loc[0]['model'].predict(test_data),1))