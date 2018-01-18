#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: tian
@contact: nieliangtian@foxmail.com
@python: python3
@software: PyCharm Community Edition
@file: stock_predict_2.py
@time: 2018/1/11 15:12
"""
import codecs
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import os

# TensorFlow日志默认显示等级，显示所有信息，'2' 只显示 warning 和 Error，'3' 只显示 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体，用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 定义常量
rnn_unit = 10       # 隐层维度
input_size = 7      # 输入层维度
output_size = 1     # 输出层维度
lr = 0.0006         # 学习率
# ——————————————————导入数据——————————————————————
f = codecs.open('dataset/dataset_2.csv', 'r', 'utf-8')
df = pd.read_csv(f)  # 读入股票数据
data = df.iloc[:, 2:10].values  # 取第3-10列


# 获取训练集
# batch_size    每一批次训练的样本数
# time_step     时间步,每次取的样本数
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=5800):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :7]
        y = normalized_train_data[i:i + time_step, 7, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(time_step=20, test_begin=5800):
    data_test = data[test_begin:]
    test_end = len(data_test) // time_step * time_step + test_begin
    data_test = data[test_begin:test_end]
    date = np.array(df['日期'])
    date = date[test_begin:test_end]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :7]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :7]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, 7]).tolist())
    return mean, std, test_x, test_y, date


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# —————————————————————定义LSTM—————————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ————————————————训练模型————————————————————

def train_lstm(batch_size=60, time_step=20, train_begin=2000, train_end=5800):

    start_time = time.time()

    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    # 这里的参数是基于已有模型恢复的参数，意思就是说之前训练过模型，保存过神经网络的参数，现在再取出来作为初始化参数接着训练。
    # 如果是第一次训练，就用sess.run(tf.global_variables_initializer())，
    # 也就不要用到 module_file = tf.train.latest_checkpoint() 和saver.store(sess, module_file)了
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        # 重复训练10000次
        for i in range(10000):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print("Number of iterations:", i, " loss:", loss_)
            if i % 500 == 0:
                print("model_save: ", saver.save(sess, 'model_2/stock_predict.ckpt'))
        print("The train has finished")

        end_time = time.time()
        run_time = end_time - start_time
        print('模型训练时间:', run_time_format(run_time))


# train_lstm()


# ————————————————预测模型————————————————————
def prediction(time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y, date = get_test_data(time_step)
    # with tf.variable_scope("sec_lstm", reuse=True):
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_2')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[7] + mean[7]
        test_predict = np.array(test_predict) * std[7] + mean[7]
        acc = 1 - np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
        print("模型准确率: %.2f%%" % (acc * 100))

        # 以折线图表示结果
        plt.figure(figsize=(16, 9))
        # line1, = plt.plot(list(range(len(test_predict))), test_predict, color='b', )
        # line2, = plt.plot(list(range(len(test_y))), test_y, color='r')

        # 生成横坐标
        x_date = [datetime.strptime(d, '%Y/%m/%d').date() for d in date]
        # 横坐标时间格式化
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        # 横坐标标签显示数量
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(16))
        # 自动旋转日期标记
        plt.gcf().autofmt_xdate()
        plt.grid(linestyle='--')
        line1, = plt.plot(x_date, test_predict, color='b', )
        line2, = plt.plot(x_date, test_y, color='r')

        plt.xlabel('日期(2014-09-03至2015-11-27)')
        plt.ylabel('股票价格(元)')
        plt.title('股票价格预测图(准确率:%.2f%%)' % (acc * 100))
        plt.legend((line1, line2), ('预测数据', '真实数据'), loc='upper left')
        plt.savefig('figure/股票价格预测图.png', dpi=300)
        plt.show()


# 模型训练时间格式化
def run_time_format(run_time):
    # 时间输出格式化
    if run_time < 60:
        run_time = round(run_time, 2)
        print(str(run_time) + ' 秒')
    elif run_time < 3600:
        run_time_minute = run_time // 60
        run_time_minute = int(run_time_minute)
        run_time_second = run_time - run_time_minute * 60
        run_time_second = round(run_time_second, 2)
        print(str(run_time_minute) + ' 分' + run_time_second + ' 秒')
    else:
        run_time_hour = run_time // 3600
        run_time_hour = int(run_time_hour)
        run_time_minute = (run_time - run_time_hour * 3600) // 60
        run_time_minute = int(run_time_minute)
        run_time_second = run_time - run_time_hour * 3600 - run_time_minute * 60
        run_time_second = round(run_time_second, 2)
        print(str(run_time_hour) + ' 小时' + str(run_time_minute) + ' 分' + run_time_second + ' 秒')


prediction()
