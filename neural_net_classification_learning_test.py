import tensorflow as tf
import numpy as np

dev_X = []
dev_Y = []

# Parameters
hidden_1_size = 100
hidden_2_size = 100
hidden_3_size = 100

with open('breast_cancer_shuffled_dev') as f:
    line = f.readline()
    while line:
        l_split = line.split()
        y = 1 if int(float(l_split[0])) == 2 else 0
        x = [int(float(t)) for t in l_split[1:]]
        dev_X.append(x)
        dev_Y.append([y])
        line = f.readline()

# tf Graph Input
X = tf.placeholder(dtype="float", shape=[None, 10])
Y = tf.placeholder(dtype="float", shape=[None, 1])

# hidden 1
W1 = tf.get_variable(name="weight1", shape=[10, hidden_1_size], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name="bias1", shape=[hidden_1_size,], initializer=tf.contrib.layers.xavier_initializer())
b_n_1 = tf.layers.batch_normalization(tf.add(tf.matmul(X, W1), b1), training=False)
a1 = tf.nn.relu(b_n_1)

# hidden 2
W2 = tf.get_variable(name="weight2", shape=[hidden_1_size, hidden_2_size], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name="bias2", shape=[hidden_2_size,], initializer=tf.contrib.layers.xavier_initializer())
b_n_2 = tf.layers.batch_normalization(tf.add(tf.matmul(a1, W2), b2), training=False)
a2 = tf.nn.relu(b_n_2)

# hidden 3
W3 = tf.get_variable(name="weight3", shape=[hidden_2_size, hidden_3_size], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name="bias3", shape=[hidden_3_size,], initializer=tf.contrib.layers.xavier_initializer())
b_n_3 = tf.layers.batch_normalization(tf.add(tf.matmul(a2, W3), b3), training=False)
a3 = tf.nn.relu(b_n_3)

# output layer
W4 = tf.get_variable(name="weight4", shape=[hidden_3_size, 1], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable(name="bias4", shape=[1,], initializer=tf.contrib.layers.xavier_initializer())
a4 = tf.nn.sigmoid(tf.add(tf.matmul(a3, W4), b4))

saver = tf.train.Saver()

with tf.Session() as sess:
        saver.restore(sess, "/Users/hzl/checkpoint/model.checkpoint")
        predictions = sess.run(a4, feed_dict={X: np.array(dev_X),
                                             Y: np.array(dev_Y)})
        print 'dev accuracy: %s.' % str(int(sum(predictions == dev_Y)) * 1.0 / len(dev_Y))
