#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: tian
@contact: nieliangtian@foxmail.com
@python: python3
@software: PyCharm Community Edition
@file: predict_tensorflow.py
@time: 2018/1/11 14:19
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# input
# 统计一天某两个时段的股票价格作为输入，然后把下一个时段初始时间的股票价格作为输出，使用BP神经网络完成
# 成交价(单位：元)
price_in_one = [30.55,
                30.74,
                30.60,
                30.90,
                30.90,
                30.90,
                30.90,
                30.96,
                30.96,
                30.95,
                30.96,
                30.97,
                30.96,
                30.80,
                30.80,
                30.80,
                30.80,
                30.70,
                30.71,
                30.72
                ]
# 成交价(单位：元)
price_in_two = [30.78,
                30.80,
                30.80,
                30.78,
                30.78,
                30.78,
                30.75,
                30.55,
                30.70,
                30.70,
                30.75,
                30.78,
                30.78,
                30.70,
                30.72,
                30.70,
                30.50,
                30.50,
                30.49,
                30.67
                ]

# output
# 成交价(单位：元)
price_out = [30.70,
             30.70,
             30.70,
             30.69,
             30.70,
             30.68,
             30.69,
             30.60,
             30.50,
             30.50,
             30.50,
             30.52,
             30.55,
             30.50,
             30.59,
             30.59,
             30.69,
             30.69,
             30.65,
             30.70
             ]


# 激活函数
# σ(x)=1/(1+e−x)
def logistic(x):
    return 1 / (1 + np.exp(-x))


# x值
# mat() 向量化 list -> mat [[...],[...],[...]], tolist() mat -> list [...],[...],[...]
sample_in = np.mat([price_in_one, price_in_two])  # 2*20 即 [[...],[...],[...]]
# h按行求最大和最小值，保存在矩阵中，再转置
sample_in_min_max = np.array(
    [sample_in.min(axis=1).T.tolist()[0], sample_in.max(axis=1).T.tolist()[0]]).transpose()  # 2*2，对应最大值最小值
# y值
sample_out = np.mat([price_out])  # 1*20
# h按行求最大和最小值，保存在矩阵中，再转置
sample_out_min_max = np.array(
    [sample_out.min(axis=1).T.tolist()[0], sample_out.max(axis=1).T.tolist()[0]]).transpose()  # 1*2，对应最大值最小值

# x归一化操作
# (X - Min) / (Max - Min)
sample_in_norm = (2 * (np.array(sample_in.T) - sample_in_min_max.transpose()[0]) / (
    sample_in_min_max.transpose()[1] - sample_in_min_max.transpose()[0]) - 1).transpose()
# y归一化操作
sample_out_norm = (2 * (np.array(sample_out.T).astype(float) - sample_out_min_max.transpose()[0]) / (
    sample_out_min_max.transpose()[1] - sample_out_min_max.transpose()[0]) - 1).transpose()

# 给输出样本添加噪音
# 人工加一些误差,噪音数据与输出数据相加,相差比例0.03
noise = 0.03 * np.random.rand(sample_out_norm.shape[0], sample_out_norm.shape[1])
sample_out_norm += noise

x = tf.placeholder(tf.float32, [None, 784])

# 实际与预测画图对比

# 仿真输出和实际输出对比图
hidden_out = logistic((np.dot(w1, sample_in_norm).transpose() + b1.transpose())).transpose()
network_out = (np.dot(w2, hidden_out).transpose() + b2.transpose()).transpose()

# 将归一化数据反算成为正常的数据
diff = sample_out_min_max[:, 1] - sample_out_min_max[:, 0]

network_out2 = (network_out + 1) / 2
network_out2[0] = network_out2[0] * diff[0] + sample_out_min_max[0][0]

sample_out = np.array(sample_out)

# 股票价格
# 创建一个 10 * 7 点的图，并设置分辨率为150
plt.figure(figsize=(8, 5), dpi=150)

# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第一块（也是唯一一块）
plt.subplot(1, 1, 1)

# 绘制
line1, = plt.plot(network_out2[0], color='b', linewidth=1, linestyle='-')
line2, = plt.plot(sample_out[0], color='r', linewidth=1, linestyle='-')

# 设置横轴上下限
plt.xlim(9.0, 15.0)

# 设置纵轴上下限
plt.ylim(30.45, 30.75)

plt.xlabel('时间/时')
plt.ylabel('股票价格/元')
plt.title('基于时间的股票价格预测模型图')
plt.legend((line1, line2), ('预测结果', '原始结果'), loc='upper left')
plt.show()
