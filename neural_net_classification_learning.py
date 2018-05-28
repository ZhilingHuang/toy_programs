import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.001
training_epochs = 100
hidden_1_size = 100
hidden_2_size = 100
hidden_3_size = 100
batch_size = 64
l2_regularization_penalty = 0.001

train_X = []
train_Y = []
with open('breast_cancer_shuffled_train') as f:
    line = f.readline()
    while line:
        l_split = line.split()
        y = 1 if int(float(l_split[0])) == 2 else 0
        x = [int(float(t)) for t in l_split[1:]]
        train_X.append(x)
        train_Y.append([y])
        line = f.readline()

# tf Graph Input
X = tf.placeholder(dtype="float", shape=[None, 10])
Y = tf.placeholder(dtype="float", shape=[None, 1])

# hidden 1
W1 = tf.get_variable(name="weight1", shape=[10, hidden_1_size], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name="bias1", shape=[hidden_1_size,], initializer=tf.contrib.layers.xavier_initializer())
b_n_1 = tf.layers.batch_normalization(tf.add(tf.matmul(X, W1), b1), training=True)
a1 = tf.nn.relu(b_n_1)

# hidden 2
W2 = tf.get_variable(name="weight2", shape=[hidden_1_size, hidden_2_size], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name="bias2", shape=[hidden_2_size,], initializer=tf.contrib.layers.xavier_initializer())
b_n_2 = tf.layers.batch_normalization(tf.add(tf.matmul(a1, W2), b2), training=True)
a2 = tf.nn.relu(b_n_2)

# hidden 3
W3 = tf.get_variable(name="weight3", shape=[hidden_2_size, hidden_3_size], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name="bias3", shape=[hidden_3_size,], initializer=tf.contrib.layers.xavier_initializer())
b_n_3 = tf.layers.batch_normalization(tf.add(tf.matmul(a2, W3), b3), training=True)
a3 = tf.nn.relu(b_n_3)

# output layer
W4 = tf.get_variable(name="weight4", shape=[hidden_3_size, 1], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable(name="bias4", shape=[1,], initializer=tf.contrib.layers.xavier_initializer())
a4 = tf.nn.sigmoid(tf.add(tf.matmul(a3, W4), b4))

# Mean squared error
cost = tf.divide(tf.reduce_sum(-Y * tf.log(a4)-(1-Y) * tf.log(1-a4)), tf.cast(tf.shape(X)[0], 'float32'))

# Gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost + l2_regularization_penalty * (
                                       tf.nn.l2_loss(W1) +
                                       tf.nn.l2_loss(W2) +
                                       tf.nn.l2_loss(W3) +
                                       tf.nn.l2_loss(W4)))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        samples = len(train_Y)
        num_batch = samples/batch_size

        train_cost = []
        for _ in range(training_epochs):
            for index in range(num_batch):
                if index == num_batch - 1:
                    c, _ = sess.run([cost, optimizer], feed_dict={X: np.array(train_X[index * batch_size:]),
                                                                  Y: np.array(train_Y[index * batch_size:])})
                else:
                    c, _ = sess.run([cost, optimizer], feed_dict={X: np.array(train_X[index*batch_size:(index+1)*batch_size]),
                                                                  Y: np.array(train_Y[index*batch_size:(index+1)*batch_size])})
                print 'batch cost: %s' % (str(c))

        save_path = saver.save(sess, "/Users/hzl/checkpoint/model.checkpoint")

