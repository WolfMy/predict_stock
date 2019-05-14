import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# 股票策略模版
# 初始化函数,全局只运行一次
def init(context):
    # 设置基准收益：沪深300指数
    set_benchmark('000300.SH')
    # 打印日志
    log.info('策略开始运行,初始化函数全局只运行一次')
    # 设置股票每笔交易的手续费为万分之二(手续费在买卖成交后扣除,不包括税费,税费在卖出成交后扣除)
    set_commission(PerShare(type='stock',cost=0.0002))
    # 设置股票交易滑点0.5%,表示买入价为实际价格乘1.005,卖出价为实际价格乘0.995
    set_slippage(PriceSlippage(0.005))
    # 设置日级最大成交比例25%,分钟级最大成交比例50%
    # 日频运行时，下单数量超过当天真实成交量25%,则全部不成交
    # 分钟频运行时，下单数量超过当前分钟真实成交量50%,则全部不成交
    set_volume_limit(0.25,0.5)
    context.num = 20
    run_monthly(func=trading_log, date_rule=-1)
    context.trade_stocks = []

## 开盘时运行函数
def handle_bar(context, bar_dict):
    cash = context.portfolio.available_cash/(context.num-len(list(context.portfolio.stock_account.positions.keys())))
    for stock in context.trade_stocks:
        bpnet = context.model_dict[stock]
        df_all = history(stock, ['close', 'prev_close', 'volume', 'turnover', 'turnover_rate'], 1, '1d')
        test_data, test_label = handle_data(df_all)
        prediction = bpnet.predict(test_data)
        if np.argmax(prediction) == 0:
            order_target_value(stock, cash)
            print(stock, '买入', cash)
        elif np.argmax(prediction) == 2:
            order_target(stock, 0)
            log.info(stock, '卖出')
        elif np.argmax(prediction) == 1:
            if stock not in list(context.portfolio.stock_account.positions.keys()):
                order_target_value(stock, cash)
                print(stock, '买入', cash)
            #else:
                #print(stock, '继续持有')
        else:
            log.info('error!')
            
def trading_log(context, bar_dict):
    # 先清空所有持仓
    if len(list(context.portfolio.stock_account.positions.keys())) > 0:
        for stock in list(context.portfolio.stock_account.positions.keys()):
            order_target(stock, 0)
    date = get_datetime().strftime('%Y-%m-%d')
    universe = list(get_all_securities('stock', date=date).index)
    df_all = history(universe, ['close', 'prev_close', 'volume', 'turnover', 'turnover_rate', 'is_st'], 60, '1d')
    context.trade_stocks = []
    context.model_dict = {}
    count = 0
    for stock in context.stock_list:
        train_data, train_label = handle_data(df_all)
        if type(train_data) != type(None):
            bpnet = BpNet(input_size=train_data.shape[1], num_classes=3, learning_rate=0.001, batch_size=32, num_epochs=1000)
            bpnet.train(train_data, train_label)
            if bpnet.get_accuracy(train_data, train_label) > 0.7:
                context.trade_stocks.append(stock)
                context.model_dict[stock] = bpnet
                count = count + 1
                log.info('{}训练完成,准确率：{}({}/{})'.format(stock, bpnet.get_accuracy(train_data, train_label), count, context.num))
                if len(context.trade_stocks) >= context.num:
                    break
    log.info('交易股票池{}'.format(context.trade_stocks))

def get_pchange_labels(data_pchange, threshold=2):
    data_labels = pd.DataFrame(columns=['buy', 'hold', 'sell'])
    for i in range(0, data_pchange.shape[0]):
        if data_pchange[i] > threshold:
            data_labels.loc[i] = [1.0, 0.0, 0.0]
        elif data_pchange[i] > -threshold:
            data_labels.loc[i] = [0.0, 1.0, 0.0]
        else:
            data_labels.loc[i] = [0.0, 0.0, 1.0]
    return data_labels

def handle_data(df_all):
    train_data = df_all[['close', 'volume', 'turnover', 'turnover_rate']].values
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)

    data_pchange = ((df_all['close'] - df_all['prev_close']) / df_all['prev_close']) * 100
    train_label = get_pchange_labels(data_pchange).values
    return train_data, train_label

################## 以下为BP模型 ##########################
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
        for i in range(self.num_epochs):
            self.sess.run (self.train_step, feed_dict = {self.x: train_data, self.y_: train_label})

    def get_accuracy(self, data, label):
        return self.sess.run(self.accuracy, feed_dict={self.x: data, self.y_: label})
    
    def predict(self, data):
        return self.sess.run(self.y, feed_dict={self.x: data})