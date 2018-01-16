# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def main(_):
    # 导入数据 one-hot code, 独热码, 数据数字化
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # 构建模型
    # 不是一个特定的值，而是一个占位符 placeholder ，我们在TensorFlow运行计算时输入这个值。
    # 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]
    # 这里的 None 表示此张量的第一个维度可以是任何长度的。
    x = tf.placeholder(tf.float32, [None, 784])

    # 模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入（使用占位符），但TensorFlow有一个更好的方法来表示它们：Variable 。
    # 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。
    # 我们赋予 tf.Variable 不同的初值来创建不同的 Variable ：在这里，我们都用全为零的张量来初始化 W 和b 。
    # 因为我们要学习 W 和 b 的值，它们的初值可以随意设置。
    # 注意， W 的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。
    # b 的形状是[10]，所以我们可以直接把它加到输出上面。
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 实现模型
    # tf.matmul(X，W) 表示 x 乘以 W
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 训练模型
    # 为了训练我们的模型，通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。
    # 一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。
    # 为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 计算交叉熵
    # y 是我们预测的概率分布, y_ 是实际的分布
    # 用 tf.log 计算 y 的每个元素的对数。接下来，我们把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。
    # 最后，用 tf.reduce_sum 计算张量的所有元素的总和。
    # 注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。
    # 对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # 交叉熵的另一种计算方法
    # y = tf.matmul(x, W) + b
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用反向传播算法(backpropagation algorithm)
    # 来有效地确定你的变量是如何影响你想要最小化的那个成本值的。
    # 然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。
    # 在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    saver = tf.train.Saver()
    # 启动并运行模型
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 训练模型
    # 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
    # 使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练
    # for _ in range(1000):
    #     batch_xs, batch_ys = mnist.train.next_batch(100)
    #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 保存模型
    # saver.save(sess, 'model/mnist.ckpt')

    # 加载模型
    saver.restore(sess, 'model/mnist.ckpt')

    # 评估模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('准确率：%.2f%%' % (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100))

    # # 参数打印
    # for var in tf.trainable_variables():
    #     print(var.eval())


if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    # 接收参数
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
