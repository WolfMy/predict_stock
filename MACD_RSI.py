import numpy as np
import pandas as pd
import tushare as ts
from sklearn.preprocessing import MinMaxScaler

#定义函数，获取macd,导入数据，初始化三个参数
def get_macd_data(data,short=0,long=0,mid=0):
    if short==0:short=12
    if long==0:long=26
    if mid==0:mid=9
    #计算短期的ema，使用pandas的ewm得到指数加权的方法，mean方法指定数据用于平均
    data['sema']=pd.Series(data['close']).ewm(span=short).mean()
    #计算长期的ema，方式同上
    data['lema']=pd.Series(data['close']).ewm(span=long).mean()
    #填充为na的数据
    data.fillna(0,inplace=True)
    #计算dif，加入新列data_dif
    data['data_dif']=data['sema']-data['lema']
    #计算dea
    data['data_dea']=pd.Series(data['data_dif']).ewm(span=mid).mean()
    #计算macd
    data['data_macd']=2*(data['data_dif']-data['data_dea'])
    #填充为na的数据
    data.fillna(0,inplace=True)
    #返回data的三个新列
    return data[['data_dif','data_dea','data_macd']]

def get_rsi_data(t, periods=10):
    length = len(t)
    rsies = [np.nan]*length
    #数据长度不超过周期，无法计算；
    if length <= periods:
        return rsies
    #用于快速计算；
    up_avg = 0
    down_avg = 0

    #首先计算第一个RSI，用前periods+1个数据，构成periods个价差序列;
    first_t = t[:periods+1]
    for i in range(1, len(first_t)):
        #价格上涨;
        if first_t[i] >= first_t[i-1]:
            up_avg += first_t[i] - first_t[i-1]
        #价格下跌;
        else:
            down_avg += first_t[i-1] - first_t[i]
    up_avg = up_avg / periods
    down_avg = down_avg / periods
    rs = up_avg / down_avg
    rsies[periods] = 100 - 100/(1+rs)

    #后面的将使用快速计算；
    for j in range(periods+1, length):
        up = 0
        down = 0
        if t[j] >= t[j-1]:
            up = t[j] - t[j-1]
            down = 0
        else:
            up = 0
            down = t[j-1] - t[j]
        #类似移动平均的计算公式;
        up_avg = (up_avg*(periods - 1) + up)/periods
        down_avg = (down_avg*(periods - 1) + down)/periods
        rs = up_avg/down_avg
        rsies[j] = 100 - 100/(1+rs)
    return rsies  

def get_shares_data(stock, start, end):
    data = pd.DataFrame(ts.get_k_data(code=stock, start=start, end=end))
    data = data.reset_index(drop=True)
    # add p_change
    p_change_list = [0]
    for i in range(1, data.shape[0]):
        last_close = data['close'][i-1]
        close = data['close'][i]
        p_change = (close - last_close)/last_close * 100
        p_change_list.append(p_change)
    data['p_change'] = p_change_list
    return data

def get_pchange_labels(data_pchange, threshold=2):
    data_labels = pd.DataFrame(columns=['buy', 'hold', 'sell'])
    for i in range(1, data_pchange.shape[0]):
        if data_pchange[i] > threshold:
            data_labels.loc[i-1] = [1.0, 0.0, 0.0]
        elif data_pchange[i] > -threshold:
            data_labels.loc[i-1] = [0.0, 1.0, 0.0]
        else:
            data_labels.loc[i-1] = [0.0, 0.0, 1.0]
    return data_labels

def init_train_data(stock, start_date, end_date):
    data = get_shares_data(stock, start_date, end_date)
    data_macd = get_macd_data(data)['data_macd']
    data_rsi = pd.DataFrame(get_rsi_data(data['close']), columns=['data_rsi'])
    data_input = pd.concat([data_macd, data_rsi], axis=1)
    data_input = data_input[10:-1].values

    data_labels = get_pchange_labels(data['p_change'])
    data_labels = data_labels[10:].values

    # 标准化
    scaler = MinMaxScaler()
    data_input = scaler.fit_transform(data_input)

    return data_input, data_labels


