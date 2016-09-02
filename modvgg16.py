import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        #self.fc8 = self.fc_layer(self.relu7, "fc8")
        self.fc8 = self.readout_layer(self.fc7, "fc8", 2)

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print("build model finished: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def readout_layer(self, bottom, name, n_class):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *=d
            x = tf.reshape(bottom, [-1, dim])
            weights = self.weight_variable([dim, n_class])
            biases = self.bias_variable([n_class])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def weight_variable(self, shape):
        # initial = tf.truncated_normal(shape, stddev=0.1)
        # return tf.Variable(initial)
        return tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape):
    #   initial = tf.constant(0.1, shape=shape)
    #   return tf.Variable(initial)
        return tf.get_variable("b", shape=shape, initializer = tf.constant_initializer(0.1))

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    #
    # def loss(logits, labels):
    #   """Calculates the loss from the logits and the labels.
    #   Args:
    #     logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    #     labels: Labels tensor, int32 - [batch_size].
    #   Returns:
    #     loss: Loss tensor of type float.
    #   """
    #   labels = tf.to_int64(labels)
    #   cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #       logits, labels, name='xentropy')
    #   loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    #   return loss
    #
    # def training(loss, learning_rate):
    #   """Sets up the training Ops.
    #   Creates a summarizer to track the loss over time in TensorBoard.
    #   Creates an optimizer and applies the gradients to all trainable variables.
    #   The Op returned by this function is what must be passed to the
    #   `sess.run()` call to cause the model to train.
    #   Args:
    #     loss: Loss tensor, from loss().
    #     learning_rate: The learning rate to use for gradient descent.
    #   Returns:
    #     train_op: The Op for training.
    #   """
    #   # Add a scalar summary for the snapshot loss.
    #   tf.scalar_summary(loss.op.name, loss)
    #   # Create the gradient descent optimizer with the given learning rate.
    #   optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #   # Create a variable to track the global step.
    #   global_step = tf.Variable(0, name='global_step', trainable=False)
    #   # Use the optimizer to apply the gradients that minimize the loss
    #   # (and also increment the global step counter) as a single training step.
    #   train_op = optimizer.minimize(loss, global_step=global_step)
    #   return train_op

# # Launch the graph in a session.
# sess = tf.Session()
# # Create a summary writer, add the 'graph' to the event file.
# logpath = '/home/geena/projects/tensorflow-vgg'
# writer = tf.train.SummaryWriter(logpath, sess.graph)

# bottom = vgg.relu7
# with tf.variable_scope('fc78'):
#     shape = bottom.get_shape().as_list()
#     dim =1
#     for d in shape[1:]:
#         dim *= d
#     x = tf.reshape(bottom,[-1,dim])
#     weights = weight_variable([dim, n_class])
#     biases = bias_variable([n_class])
#     fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
