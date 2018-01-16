import tensorflow as tf

# 第一个TensorFlow程序，HelloWorld


def sayHello(words):
    hello = tf.constant(words)
    session = tf.Session()
    print(session.run(hello))


if __name__ == '__main__':
    words = "Hello world"
    sayHello(words)
