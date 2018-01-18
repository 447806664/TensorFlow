#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: tian
@contact: nieliangtian@foxmail.com
@python: python3
@software: PyCharm Community Edition
@file: predict_bp_nn2.py
@time: 2018/1/10 14:47
"""

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

# 训练参数设置
max_epochs = 10000  # 最大迭代次数
learn_rate = 0.035  # 学习速率
error_final = 0.65 * 10 ** (-3)  # 允许误差
sample_num = 20
in_dimension_num = 2  # 输入层数量
out_dimension_num = 1  # 输出层数量
hidden_unit_num = 5  # 隐含层数量

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
    plt.plot(error_history10, linewidth=0.5)

    plt.xlabel('迭代次数')
    plt.ylabel('误差平方和')
    plt.title('模型训练误差图')

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
