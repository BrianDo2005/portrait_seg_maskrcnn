import tensorflow as tf



class Conv2dSobel():
    # Sobel arithmetic operators

    def __init__(self, name):
        self.name = name

    def SobelX(self, input):
        W = tf.constant([-1,-2,-1,0,0,0,1,2,1], dtype=tf.float32, shape=[3,3,1,1])

        output = tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME')

        return output

    def SobelY(self, input):
        W = tf.constant([-1,0,1,-2,0,2,-1,0,1], dtype=tf.float32, shape=[3,3,1,1])

        output = tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME')

        return output

