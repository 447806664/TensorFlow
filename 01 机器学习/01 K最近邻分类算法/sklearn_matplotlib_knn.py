#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
'''用来正常显示中文标签'''
plt.rcParams['font.sans-serif'] = ['SimHei']
'''用来正常显示负号'''
plt.rcParams['axes.unicode_minus'] = False

'''数据读取'''
data = []
labels = []
with open("data/build.txt") as file:
    for line in file:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)

'''标签转换为0/1'''
y[labels == 'fat'] = 1

'''拆分训练数据与测试数据'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

'''创建网格以方便绘制'''
h = .01
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

'''
KNN分类器
KNeighborsClassifier(n_neighbors=5, weights='uniform', 
                     algorithm='auto', leaf_size=30, 
                     p=2, metric='minkowski', 
                     metric_params=None, n_jobs=1, **kwargs)
n_neighbors: 默认值为5，表示查询k个最近邻的数目
algorithm:   {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’},指定用于计算最近邻的算法，auto表示试图采用最适合的算法计算最近邻
leaf_size:   传递给‘ball_tree’或‘kd_tree’的叶子大小
metric:      用于树的距离度量。默认'minkowski与P = 2（即欧氏度量）
n_jobs:      并行工作的数量，如果设为-1，则作业的数量被设置为CPU内核的数量
查看官方api：http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
'''
clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
# 训练数据集
clf.fit(x_train, y_train)

'''测试结果的打印'''
answer = clf.predict(x)
print(x)
print(answer)
print(y)
print(np.mean(answer == y))

'''准确率与召回率'''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
answer = clf.predict(x)
print(classification_report(y, answer, target_names=['thin', 'fat']))

'''将整个测试空间的分类结果用不同颜色区分开'''
answer = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
z = answer.reshape(xx.shape)
plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

'''绘制训练样本'''
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.xlabel(u'身高')
plt.ylabel(u'体重')
plt.title(u'KNN演示')
plt.show()
