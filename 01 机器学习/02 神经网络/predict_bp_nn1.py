#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: tian
@contact: nieliangtian@foxmail.com
@python: python3
@software: PyCharm Community Edition
@file: predict_bp_nn1.py
@time: 2018/1/10 14:47
"""

import numpy as np
import matplotlib.pyplot as plt

# 1990-2009 20年的人口数、车辆数、公路面积作为输入，然后把公路客运量和公路货运量作为输出，使用BP神经网络完成
# input 1990-2009
# 人数(单位：万人)
population = [20.55, 22.44, 25.37, 27.13, 29.45, 30.10, 30.96, 34.06, 36.42, 38.09, 39.13, 39.99, 41.93, 44.59, 47.30,
              52.89, 55.73, 56.76, 59.17, 60.63]
# 机动车数(单位：万辆)
vehicle = [0.6, 0.75, 0.85, 0.9, 1.05, 1.35, 1.45, 1.6, 1.7, 1.85, 2.15, 2.2, 2.25, 2.35, 2.5, 2.6, 2.7, 2.85, 2.95,
           3.1]
# 公路面积(单位：万平方公里)
road_area = [0.09, 0.11, 0.11, 0.14, 0.20, 0.23, 0.23, 0.32, 0.32, 0.34, 0.36, 0.36, 0.38, 0.49, 0.56, 0.59, 0.59,
             0.67, 0.69, 0.79]

# output
# 公路客运量(单位：万人)
passenger_traffic = [5126, 6217, 7730, 9145, 10460, 11387, 12353, 15750, 18304, 19836, 21024, 19490, 20433, 22598,
                     25107, 33442, 36836, 40548, 42927, 43462]
# 公路货运量(单位：万吨)
freight_traffic = [1237, 1379, 1385, 1399, 1663, 1714, 1834, 4322, 8132, 8936, 11099, 11203, 10524, 11115, 13320,
                   16762, 18673, 20724, 20803, 21804]


# 激活函数
# σ(x)=1/(1+e−x)
def logistic(x):
    return 1 / (1 + np.exp(-x))


# x值
# mat() 向量化 list -> mat [[...],[...],[...]], tolist() mat -> list [...],[...],[...]
sample_in = np.mat([population, vehicle, road_area])  # 3*20 即 [[...],[...],[...]]
# h按行求最大和最小值，保存在矩阵中，再转置
sample_in_min_max = np.array(
    [sample_in.min(axis=1).T.tolist()[0], sample_in.max(axis=1).T.tolist()[0]]).transpose()  # 3*2，对应最大值最小值
# y值
sample_out = np.mat([passenger_traffic, freight_traffic])  # 2*20
# h按行求最大和最小值，保存在矩阵中，再转置
sample_out_min_max = np.array(
    [sample_out.min(axis=1).T.tolist()[0], sample_out.max(axis=1).T.tolist()[0]]).transpose()  # 2*2，对应最大值最小值

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

# 训练参数设置
max_epochs = 1000  # 最大迭代次数
learn_rate = 0.035  # 学习速率
error_final = 0.65 * 10 ** (-3)  # 允许误差
sample_num = 20
in_dimension_num = 3  # 输入层数量
out_dimension_num = 2  # 输出层数量
hidden_unit_num = 8  # 隐含层数量

w1 = 0.5 * np.random.rand(hidden_unit_num, in_dimension_num) - 0.1  # 输入层系数（线上）
b1 = 0.5 * np.random.rand(hidden_unit_num, 1) - 0.1  # 隐含层阈值
w2 = 0.5 * np.random.rand(out_dimension_num, hidden_unit_num) - 0.1  # 输出层系数（线上）
b2 = 0.5 * np.random.rand(out_dimension_num, 1) - 0.1  # 输出层阈值

# 用于存储每一次的误差平方和
error_history = []

# BP正向传递
# max_epochs 迭代次数
for i in range(max_epochs):
    # fp
    # 隐含层计算值
    hidden_out = logistic((np.dot(w1, sample_in_norm).transpose() + b1.transpose())).transpose()

    # 输出值
    network_out = (np.dot(w2, hidden_out).transpose() + b2.transpose()).transpose()

    # 误差
    error = sample_out_norm - network_out

    # 误差平方和
    square_sum = sum(sum(error ** 2))

    # 误差平方和集合，后面画图观察误差下降使用
    error_history.append(square_sum)

    # 误差满足要求停止计算
    if square_sum < error_final:
        break

    delta2 = error

    # BP误差反向传递
    # 这个是公式推导出来的：(传递到隐含层线上)
    # 输出层 x 误差 x 隐含层节点计算值 x (1 - 隐含层节点计算值)
    delta1 = np.dot(w2.transpose(), delta2) * hidden_out * (1 - hidden_out)

    # 误差 x 隐含层节点计算值
    dw2 = np.dot(delta2, hidden_out.transpose())  # 权值
    db2 = np.dot(delta2, np.ones((sample_num, 1)))  # 阈值

    # 继续反向传递:(传递到输入层线上)
    # b = 输出层 x 误差 x 隐含层节点计算值 x (1 - 隐含层节点计算值)
    # b x 输入层数据y
    dw1 = np.dot(delta1, sample_in_norm.transpose())  # 权值
    db1 = np.dot(delta1, np.ones((sample_num, 1)))  # 阈值

    # 通过一定的步长去更新阈值和权值
    w2 += learn_rate * dw2
    b2 += learn_rate * db2

    w1 += learn_rate * dw1
    b1 += learn_rate * db1

    print("BP神经网络训练得到参数如下：")
    print("输入层权值:" + str(w1))
    print("输入层阈值:" + str(b1))
    print("隐含层权值:" + str(w2))
    print("隐含层阈值:" + str(b2))

    # 误差曲线图
    error_history10 = np.log10(error_history)
    min_error = min(error_history10)
    plt.plot(error_history10)
    plt.plot(range(0, i + 1000, 1000), [min_error] * len(range(0, i + 1000, 1000)))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    ax = plt.gca()
    ax.set_yticks([-2, -1, 0, 1, 2, min_error])
    ax.set_yticklabels([u'$10^{-2}$', u'$10^{-1}$', u'$1$', u'$10^{1}$', u'$10^{2}$',
                        str(('%.4f' % np.power(10, min_error)))])
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('误差平方和log10')
    ax.set_title('模型训练误差图')

# #################################实际与预测画图对比#################################

# 仿真输出和实际输出对比图
hidden_out = logistic((np.dot(w1, sample_in_norm).transpose() + b1.transpose())).transpose()
network_out = (np.dot(w2, hidden_out).transpose() + b2.transpose()).transpose()

# 将归一化数据反算成为正常的数据
diff = sample_out_min_max[:, 1] - sample_out_min_max[:, 0]

network_out2 = (network_out + 1) / 2
network_out2[0] = network_out2[0] * diff[0] + sample_out_min_max[0][0]
network_out2[1] = network_out2[1] * diff[1] + sample_out_min_max[1][0]

sample_out = np.array(sample_out)
# 客运量
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
line1, = axes[0].plot(network_out2[0], 'k', marker=u'$\circ$')  # 预测结果
line2, = axes[0].plot(sample_out[0], 'r', markeredgecolor='b', marker=u'$\star$', markersize=9)  # 原始结果

axes[0].legend((line1, line2), ('预测结果', '原始结果'), loc='upper left')

y_ticks = [0, 20000, 40000, 60000]
y_ticks_label = [u'$0$', u'$2$', u'$4$', u'$6$']
axes[0].set_yticks(y_ticks)
axes[0].set_yticklabels(y_ticks_label)
axes[0].set_ylabel(u'客运量$(10^4)$')

x_ticks = range(0, 20, 2)
x_ticks_label = range(1990, 2010, 2)
axes[0].set_xticks(x_ticks)
axes[0].set_xticklabels(x_ticks_label)
axes[0].set_xlabel(u'年份')
axes[0].set_title('基于时间的客运量预测模型图')

# 货运量
line3, = axes[1].plot(network_out2[1], 'k', marker=u'$\circ$')
line4, = axes[1].plot(sample_out[1], 'r', markeredgecolor='b', marker=u'$\star$', markersize=9)
axes[1].legend((line3, line4), ('预测结果', '原始结果'), loc='upper left')
y_ticks = [0, 10000, 20000, 30000]
y_ticks_label = [u'$0$', u'$1$', u'$2$', u'$3$']
axes[1].set_yticks(y_ticks)
axes[1].set_yticklabels(y_ticks_label)
axes[1].set_ylabel(u'货运量$(10^4)$')

x_ticks = range(0, 20, 2)
x_ticks_label = range(1990, 2010, 2)
axes[1].set_xticks(x_ticks)
axes[1].set_xticklabels(x_ticks_label)
axes[1].set_xlabel(u'年份')
axes[1].set_title('基于时间的货运量预测模型图')

plt.show()
