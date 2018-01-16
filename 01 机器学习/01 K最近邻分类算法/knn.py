#!/usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import *
import operator
from os import listdir


def createDataSet():
    """
    构建一组训练数据（训练样本），共4个样本
    同时给出了这4个样本的标签，即labels
    样本与标签一一对应
    """
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0., 0.],
        [0., 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 简单分类
def classify(inX, dataset, labels, k):
    """
    inX 是输入的测试样本，是一个[x, y]样式的
    dataset 是训练样本集
    labels 是训练样本标签
    k 是top k最相近的
    """
    # 由m*n个数排成的m行n列的数表称为m行n列的矩阵，简称m × n矩阵
    # shape返回矩阵的[行数，列数]
    # 那么shape[0]获取数据集的行数
    # 行数就是样本的数量
    dataSetSize = dataset.shape[0]

    """
    下面的求距离过程就是按照欧氏距离的公式计算的
    即 根号(x^2+y^2)
    """
    # tile属于numpy模块下的函数
    # tile(A, reps)返回一个shape=reps的矩阵，矩阵的每个元素是A
    # 比如 A=[0,1,2] 那么，tile(A, 2)= [0, 1, 2, 0, 1, 2]
    # tile(A,(2,2)) = [[0, 1, 2, 0, 1, 2],
    #                  [0, 1, 2, 0, 1, 2]]
    # tile(A,(2,1,2)) = [[[0, 1, 2, 0, 1, 2]],
    #                    [[0, 1, 2, 0, 1, 2]]]
    # 上边那个结果的分开理解就是：
    # 最外层是2个元素，即最外边的[]中包含2个元素，类似于[C,D],而此处的C=D，因为是复制出来的
    # 然后C包含1个元素，即C=[E],同理D=[E]
    # 最后E包含2个元素，即E=[F,G],此处F=G，因为是复制出来的
    # F就是A了，基础元素
    # 综合起来就是(2,1,2)= [C, C] = [[E], [E]] = [[[F, F]], [[F, F]]] = [[[A, A]], [[A, A]]]
    # 这个地方就是为了把输入的测试样本扩展为和dataset的shape一样，然后就可以直接做矩阵减法了。
    # 比如，dataset有4个样本，就是4*2的矩阵，输入测试样本肯定是一个了，就是1*2，为了计算输入样本与训练样本的距离
    # 那么，需要对这个数据进行作差。这是一次比较，因为训练样本有n个，那么就要进行n次比较；
    # 为了方便计算，把输入样本复制n次，然后直接与训练样本作矩阵差运算，就可以一次性比较了n个样本。
    # 比如inX = [0,1],dataset就用函数返回的结果，那么
    # tile(inX, (4,1))= [[ 0.0, 1.0],
    #                    [ 0.0, 1.0],
    #                    [ 0.0, 1.0],
    #                    [ 0.0, 1.0]]
    # 作差之后
    # diffMat = [[-1.0,-0.1],
    #            [-1.0, 0.0],
    #            [ 0.0, 1.0],
    #            [ 0.0, 0.9]]
    diffMat = tile(inX, (dataSetSize, 1)) - dataset

    # diffMat就是输入样本与每个训练样本的差值，然后对其每个x和y的差值进行平方运算。
    # diffMat是一个矩阵，矩阵**2表示对矩阵中的每个元素进行**2操作，即平方。
    # sqDiffMat = [[1.0, 0.01],
    #              [1.0, 0.0 ],
    #              [0.0, 1.0 ],
    #              [0.0, 0.81]]
    sqDiffMat = diffMat ** 2

    # axis=1表示按照横轴，sum表示累加，即按照行进行累加
    # sum(axis=某个维) 某个维就消失，其他维不变，对消失的维求和
    # sqDistance = [[1.01],
    #               [1.0 ],
    #               [1.0 ],
    #               [0.81]]
    sqDistance = sqDiffMat.sum(axis=1)

    # 对平方和进行开根号
    distance = sqDistance ** 0.5

    # 按照升序进行快速排序，返回的是原数组的下标
    # 比如，x = [30, 10, 20, 40]
    # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
    # 那么，numpy.argsort(x) = [1, 2, 0, 3]
    sortedDistIndicies = distance.argsort()

    # 存放最终的分类结果及相应的结果投票数
    # classCount 字典
    classCount = {}

    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):
        # index = sortedDistIndicies[i]是第i个最相近的样本下标
        # voteIlabel = labels[index]是样本index对应的分类结果('A' or 'B')
        voteIlabel = labels[sortedDistIndicies[i]]
        # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        # 然后将票数增1
        # 往字典classCount里添加元素
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 把分类结果进行排序，然后返回得票数最多的分类结果
    # sorted(iterable[, cmp[, key[, reverse]]])
    # 参数解释：
    # 1) iterable指定要排序的list或者iterable；
    # 2) cmp为函数，指定排序时进行比较的函数，可以指定一个函数或者lambda函数，如：
    #   students为类对象的list，每个成员有三个域，用sorted进行比较时可以自己定cmp函数，例如这里要通过比较第三个数据成员来排序，代码可以这样写：
    #   students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
    #   sorted(students, key=lambda student: student[2])
    # 3) key为函数，指定取待排序元素的哪一项进行排序，函数用上面的例子来说明，代码如下：
    #   sorted(students, key=lambda student: student[2])
    #   key指定的lambda函数功能是去元素student的第三个域（即：student[2]），因此sorted排序时，会以students所有元素的第三个域来进行排序。
    #   也可使用operator.itemgetter函数
    #   sorted(students, key=operator.itemgetter(2))
    #   sorted函数也可以进行多级排序，例如要根据第二个域和第三个域进行排序，可以这么写：
    #   sorted(students, key=operator.itemgetter(1, 2))
    #   即先根据第二个域排序，再根据第三个域排序。
    # 4) reverse参数，是一个bool变量，表示升序还是降序排列，默认为false（升序排列），定义为True时将按降序排列。
    # classCount.keys()表示字典的键，classCount.items()表示字典的值，此处字典元素的形式为 voteIlabel:num
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 在约会网站上使用KNN
# datingTestSet.txt 文件中有1000行的约会数据，样本主要包括以下3种特征：
#     每年获得的飞行常客里程数
#     玩视频游戏所耗时间百分比
#     每周消费的冰淇淋公升数
# 将上述特征数据输人到分类器之前，必须将待处理数据的格式改变为分类器可以接受的格式 。
# 创建名为 file2matrix 的函数，以此来处理输人格式问题。该函数的输入为文件名字符串，输出为训练样本矩阵和类标签向量。
# autoNorm 为数值归一化函数，将任意取值范围的特征值转化为0到1区间内的值。
# 最后，datingClassTest 函数是测试代码。
def file2matrix(filename):
    """
    从文件中读取训练数据，并存储为矩阵
    """
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)  # 获取 n=样本的行数
    returnMat = zeros((numberOfLines, 3))  # 创建一个2维矩阵用于存放训练样本数据，一共有n行，每一行存放3个数据
    classLabelVector = []  # 创建一个1维数组用于存放训练样本标签。
    index = 0
    for line in arrayLines:
        # 把回车符号给去掉
        line = line.strip()
        # 把每一行数据用\t分割
        listFromLine = line.split('\t')
        # 把分割好的数据放至数据集，其中index是该样本数据的下标，就是放到第几行
        # 如下是把数据从index开始一直到放完
        returnMat[index, :] = listFromLine[0:3]
        # 把该样本对应的标签放至标签集，顺序与样本集对应
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    训练数据归一化
    """
    # 获取数据集中每一列的最小数值
    # 以createDataSet()中的数据为例，group.min(0)=[0,0]
    minVals = dataSet.min(0)
    # 获取数据集中每一列的最大数值
    # group.max(0)=[1, 1.1]
    maxVals = dataSet.max(0)
    # 最大值与最小的差值
    ranges = maxVals - minVals
    # 创建一个与dataSet同shape的全0矩阵，用于存放归一化后的数据
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 把最小值扩充为与dataSet同shape，然后作差
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 把最大最小差值扩充为dataSet同shape，然后作商，是指对应元素进行除法运算，而不是矩阵除法。
    # 矩阵除法在numpy中要用linalg.solve(A,B)
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    # 将数据集中10%的数据留作测试用，其余的90%用于训练
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('data/datingTestSet.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("分类器分类结果: %s, 真实答案是: %s, 准确与否: %s" % (
            classifierResult, datingLabels[i], classifierResult == datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    accuracy_rate = 1 - errorCount / float(numTestVecs)
    print("\n准确率是: %.2f" % (accuracy_rate * 100) + '%')


# 为了简单起见，这里构造的系统只能识别数字0到9。
# 需要识别的数字已经使用图形处理软件，
# 处理成具有相同的色彩和大小:
# 宽髙是32像素 x 32像素的黑白图像。
# 尽管采用文本格式存储图像不能有效地利用内存空间，
# 但是为了方便理解，我们还是将图像转换为文本格式。
# trainingDigits是2000个训练样本，testDigits是900个测试样本。
def img2vector(filename):
    """
    将图片数据转换为01矩阵。
    每张图片是32*32像素，也就是一共1024个字节。
    因此转换的时候，每行表示一个样本，每个样本含1024个字节。
    """
    # 每个样本数据是1024=32*32个字节
    returnVect = zeros((1, 1024))
    fr = open(filename)
    # 循环读取32行，32列。
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 加载训练数据
    trainingFileList = listdir('data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件名中解析出当前图像的标签，也就是数字是几
        # 文件名格式为 0_3.txt 表示图片数字是 0
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('data/trainingDigits/%s' % fileNameStr)
    # 加载测试数据
    testFileList = listdir('data/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类器识别结果: %d, 真实答案是: %d, 准确与否: %s" % (
            classifierResult, classNumStr, classifierResult == classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    accuracy_rate = 1 - errorCount / float(mTest)
    print("\n准确率是: %.2f" % (accuracy_rate * 100) + '%')


# 测试
if __name__ == "__main__":
    """
    简单分类
    """
    # dataset, labels = createDataSet()
    # inX = [0.1, 0.1]
    # className = classify(inX, dataset, labels, 3)
    # print('测试样本标签是 %s' % className)

    """
    分类约会网站实例
    """
    # datingClassTest()

    """
    识别手写字体
    """
    # handwritingClassTest()
