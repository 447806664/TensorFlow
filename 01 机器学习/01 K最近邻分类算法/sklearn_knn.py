#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split

'''数据读取'''
data = []
labels = []
with open("data/iris.txt") as file:
    for line in file:
        tokens = line.strip().split(',')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
x = np.array(data)
y = np.array(labels)

'''拆分训练数据与测试数据'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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
knn = neighbors.KNeighborsClassifier()
# 训练数据集
knn.fit(x_train, y_train)

'''测试结果的打印'''
predict = knn.predict(x)
for i in range(len(predict)):
    print('分类器分类结果：%s, 真实答案是：%s, 准确与否：%s' % (predict[i], y[i], predict[i] == y[i]))
# 准确率
# np.mean() 取平均值，其中True=1,False=0
# accuracy_rate = float(np.mean(predict == y))
accuracy_rate = knn.score(x_train, y_train)
print('\n准确率：%.2f%%' % (accuracy_rate * 100))
