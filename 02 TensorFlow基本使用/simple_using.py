# 基本使用

# 使用图 (graph) 来表示计算任务.
# 在被称之为 会话 (Session) 的上下文 (context) 中执行图.
# 使用 tensor 表示数据.
# 通过 变量 (Variable) 维护状态.
# 使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.

import tensorflow as tf

# 创建一个常量op,产生一个1X2矩阵（matrix），此op被称作一个节点
# 构造器的返回值代表该常量op的返回值
matrix1 = tf.constant([[3., 3.]])
# 创建另一个常量op，产生一个2X1矩阵
matrix2 = tf.constant([[2.], [2.]])
# 创建一个矩阵乘法matmul op,把 matrix1 和 matrix2 作为输入
# 返回值product代表矩阵乘法的结果
product = tf.matmul(matrix1, matrix2)

# 默认图现在有三个节点，两个constant() op, 和一个matmul() op
# 为了真正进行矩阵相乘运算，并得到矩阵乘法的结果，必须先在会话里启动这个图
# 启动默认图
sess = tf.Session()

# 调用sess的run()方法来执行矩阵乘法op,传入product作为参数
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入，op 通常是并发执行的
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行
# 返回值 'result' 是一个 numpy `ndarray` 对象
result = sess.run(product)
print(result)

# 关闭会话
sess.close()

# Session对象在使用完后需要关闭以释放资源,除了显式调用close()外,
# 也可以使用 "with" 代码块 来自动完成关闭动作
with tf.Session() as sess2:
    result2 = sess2.run(product)
    print(result)
