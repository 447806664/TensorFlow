#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: tian
@contact: nieliangtian@foxmail.com
@python: python3
@software: PyCharm Community Edition
@file: stock_predict_3.py
@time: 2018/1/18 5:50
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

# TensorFlow日志默认显示等级
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体，用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 定义常量
rnn_unit = 5       # 隐层维度
input_size = 1      # 输入层维度
output_size = 1     # 输出层维度
lr = 0.0005         # 学习率
# ——————————————————导入数据——————————————————————
f = codecs.open('dataset/dataset_3.tsv', 'r', 'utf-8')
df = pd.read_table(f)  # 读入股票数据
data = df.iloc[:, 3:4].values  # 取第3列


# 获取训练集
# batch_size    每一批次训练的样本数
# time_step     时间步,每次取的样本数
def get_train_data(batch_size=40, time_step=10, train_begin=0, train_end=200):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step - 1):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :input_size]
        y = normalized_train_data[i + 1:i + 1 + time_step, :input_size]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(time_step=20, test_begin=200):
    data_test = data[test_begin:]
    test_end = len(data_test) // time_step * time_step + test_begin
    data_test = data[test_begin:test_end]
    date = df.iloc[:, 1:2].values.tolist()  # 取第二列
    date = np.reshape(date, [-1, ])
    date = date[test_begin:test_end]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :input_size]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, :input_size]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :input_size]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, :input_size]).tolist())
    return mean, std, test_x, test_y, date


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, output_size]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
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

def train_lstm(batch_size=40, time_step=10, train_begin=0, train_end=200):

    start_time = time.time()

    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练
        for i in range(2000):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print("迭代次数:", i, " 成本:", loss_)
            if i % 50 == 0:
                print("模型保存到: ", saver.save(sess, 'model_3/stock_predict.ckpt'))
        print("模型训练已完成！")


# train_lstm()


# ————————————————预测模型————————————————————
def prediction(time_step=10):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y, date = get_test_data(time_step)
    # with tf.variable_scope("sec_lstm", reuse=True):
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_3')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[0] + mean[0]
        test_predict = np.array(test_predict) * std[0] + mean[0]
        acc = 1 - np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
        print("模型准确率: %.2f%%" % (acc * 100))

        # 以折线图表示结果
        plt.figure(figsize=(16, 9))

        # 生成横坐标
        x_date = [datetime.strptime(str(d), '%Y%m%d').date() for d in date]
        # 横坐标时间格式化
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        # 横坐标标签显示数量
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(16))
        # 自动旋转日期标记
        plt.gcf().autofmt_xdate()
        plt.grid(linestyle='--')
        line1, = plt.plot(x_date, test_predict, color='b', )
        line2, = plt.plot(x_date, test_y, color='r')

        plt.xlabel('交易日期')
        plt.ylabel('股票价格(元)')
        plt.title('股票价格预测图(准确率:%.2f%%)' % (acc * 100))
        plt.legend((line1, line2), ('预测数据', '真实数据'), loc='upper left')
        plt.savefig('figure/股票价格预测图.png', dpi=300)
        plt.show()


prediction()
