import tensorflow as tf
import numpy as np
import pandas as pd
from MACD_RSI import init_train_data

def get_batch(data, label, batch_size, num_epochs):
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=num_epochs, shuffle=True, capacity=32)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
    return x_batch, y_batch

def BP(data_train, label_train, input_size, num_classes, learning_rate=0.001, batch_size=64, num_epochs=1000):
    X = tf.placeholder(tf.float32, shape=[None, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    W1 = tf.Variable (tf.random_uniform([input_size,10], 0,1))
    B1 = tf.Variable (tf.zeros([1, 10]))
    hidden_y1 = tf.nn.relu (tf.matmul(X, W1) + B1)

    W2 = tf.Variable (tf.random_uniform([10,7], 0,1))
    B2 = tf.Variable (tf.zeros([1, 7]))
    hidden_y2 = tf.nn.relu (tf.matmul(hidden_y1, W2) + B2)

    W3 = tf.Variable (tf.random_uniform([7, num_classes], 0.1))
    B3 = tf.Variable (tf.zeros([1, num_classes]))
    final_opt = tf.nn.softmax(tf.matmul(hidden_y2, W3) + B3)

    loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (labels = Y, logits = final_opt))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(final_opt,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    x_batch, y_batch = get_batch(data_train, label_train, batch_size, num_epochs)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # 开启协调器
        coord = tf.train.Coordinator()
        # 使用start_queue_runners 启动队列填充
        threads = tf.train.start_queue_runners(sess, coord)
        epoch = 0
        try:
            while not coord.should_stop():
                # 获取训练用的每一个batch中batch_size个样本和标签
                batch_input, batch_label = sess.run([x_batch, y_batch])
                sess.run (train_step, feed_dict = {X: batch_input, Y: batch_label})
                if epoch % 200 == 0 :
                    train_accuracy = sess.run(accuracy, feed_dict = {X: batch_input, Y: batch_label})
                    #test_accuracy = sess.run(accuracy, feed_dict = {X: data_test, Y: label_test})
                    #print ("step : %d, training accuracy = %g, test_accuracy = %g " % (epoch, train_accuracy, test_accuracy))
                    print ("step : %d, training accuracy = %g " % (epoch, train_accuracy))
                    print("loss:", sess.run(loss, feed_dict={X: batch_input, Y: batch_label}))
                epoch = epoch + 1
        except tf.errors.OutOfRangeError:  # num_epochs 次数用完会抛出此异常
            print("---Train end---")
        finally:
            # 协调器coord发出所有线程终止信号
            coord.request_stop()
            coord.join(threads)  # 把开启的线程加入主线程，等待threads结束
            print('---Programm end---')
        
        # 训练完成后，记录test_accuracy，返回[stock,test_accuracy]
            train_accuracy = sess.run(accuracy, feed_dict = {X: batch_input, Y: batch_label})
    
    return train_accuracy

start_date = '2018-11-20'
end_date = '2019-03-01'
stock_list = ['603000', '002230', '300492', '601688']
df = pd.DataFrame(stock_list, columns=['stock'])
train_acc = []
for stock in stock_list:
    data, label = init_train_data(stock, start_date, end_date)
    train_acc.append(BP(data, label, input_size=2, num_classes=3, learning_rate=0.001, batch_size=32))
df['train_acc'] = train_acc
print(df.sort_values(['train_acc'], ascending=False))