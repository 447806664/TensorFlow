#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: tian
@contact: nieliangtian@foxmail.com
@python: python3
@software: PyCharm Community Edition
@file: stock_predict_1.py
@time: 2018/1/11 15:12
"""
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# TensorFlow日志默认显示等级，显示所有信息，'2' 只显示 warning 和 Error，'3' 只显示 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# ——————————————————导入数据——————————————————————
f = codecs.open('dataset/dataset_1.csv', 'r', 'utf-8')
# 读取股票数据
df = pd.read_csv(f)
# df['最高价']取出最高价这一列的数据
data = np.array(df['最高价'])
# [::-1]反序，也可指定反序的范围，例如：[2:0:-1]
# 将列表a倒序处理，如果a＝［1，2，3］，则a［：：－1］＝［3，2，1］。前两个冒号表示处理整个列表，也可以写上参数表示处理列表的一部分，
# 例如a［2:0:－1］=［3,2］，第一个参数表示起始点包括起始点，第二个参数表示结束点但不包括结束点。最后一个参数如果为负的话，
# 需要保证第一个参数大于第二个参数，表示依次递减逆序，否则会输出空列表。最后一个参数为正同理。
data = data[::-1]

# 以折线图展示data
# plt.figure()
# plt.plot(data)
# plt.title('股票最高价变化图')
# plt.show()

# 标准化
# 两种方法：
# 1 (x-min)/(max-min) 归一(0,1)
# 2 (x-均值mean)/标准差standard deviation
normalize_data = (data - np.mean(data)) / np.std(data)
# np.newaxis 在使用和功能上等价于 None，其实就是 None 的一个别名,为 numpy.ndarray（多维数组）增加一个轴
# 增加维度
normalize_data = normalize_data[:, np.newaxis]

# 生成训练集
# 设置常量
time_step = 20      # 时间步
rnn_unit = 10       # 隐藏层维度
batch_size = 60     # 每一批次训练的样本数
input_size = 1      # 输入层维度
output_size = 1     # 输出层维度
lr = 0.0006         # 学习率
train_x, train_y = [], []   # 训练集 shape为[1,time_step,input__size]的矩阵
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())


# ——————————————————定义神经网络变量——————————————————
X = tf.placeholder(tf.float32, [None, time_step, input_size])  # 每批次输入网络的tensor
Y = tf.placeholder(tf.float32, [None, time_step, output_size])  # 每批次tensor对应的标签

# 输入层、输出层权重、偏置
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
# shape=[2,] 一维
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义LSTM——————————————————
def lstm(batch):  # 参数：输入网络批次数目
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练1000次
        for i in range(1000):
            step = 0
            start = 0
            end = start + batch_size
            while end < len(train_x):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size
                # 每100步保存一次参数
                if step % 100 == 0:
                    print("Number of iterations:", i, " loss:", loss_)
                    print("model_save", saver.save(sess, 'model_1/stock_predict.ckpt'))
                step += 1
        print("The train has finished")


# train_lstm()


# ————————————————预测模型————————————————————
def prediction():
    # with tf.variable_scope("sec_lstm", reuse=True):
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(1)  # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_1')
        saver.restore(sess, module_file)
        # 取训练集最后一行为测试样本 shape=[1,time_step,input_size]
        prev_seq = train_x[-1]
        predict = []
        # 得到之后100个预测结果
        for i in range(100):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()


prediction()
