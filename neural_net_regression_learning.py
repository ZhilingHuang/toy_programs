import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.001
training_epochs = 100
hidden_1_size = 80
hidden_2_size = 60
hidden_3_size = 40
batch_size = 64
l2_regularization_penalty = 0.0

# tf Graph Input
X = tf.placeholder(dtype="float", shape=[None, 10])
Y = tf.placeholder(dtype="float", shape=[None, 1])

# hidden 1
W1 = tf.get_variable(name="weight1", shape=[10, hidden_1_size], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name="bias1", shape=[hidden_1_size,], initializer=tf.contrib.layers.xavier_initializer())
a1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

# hidden 2
W2 = tf.get_variable(name="weight2", shape=[hidden_1_size, hidden_2_size], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name="bias2", shape=[hidden_2_size,], initializer=tf.contrib.layers.xavier_initializer())
a2 = tf.nn.leaky_relu(tf.add(tf.matmul(a1, W2), b2))

# hidden 3
W3 = tf.get_variable(name="weight3", shape=[hidden_2_size, hidden_3_size], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name="bias3", shape=[hidden_3_size,], initializer=tf.contrib.layers.xavier_initializer())
a3 = tf.nn.leaky_relu(tf.add(tf.matmul(a2, W3), b3))

# output layer
W4 = tf.get_variable(name="weight4", shape=[hidden_3_size, 1], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable(name="bias4", shape=[1,], initializer=tf.contrib.layers.xavier_initializer())
a4 = tf.add(tf.matmul(a3, W4), b4)

# Mean squared error
cost = tf.divide(tf.reduce_sum(tf.pow(a4-Y, 2)), tf.cast(tf.shape(X)[0], 'float32'))

# Gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost + l2_regularization_penalty * (
                                       tf.nn.l2_loss(W1) +
                                       tf.nn.l2_loss(W2) +
                                       tf.nn.l2_loss(W3) +
                                       tf.nn.l2_loss(W4)))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Get dev dataset.
with open('abalone_data_shuffled_dev') as f:
    line = f.readline()
    dev_X = []
    dev_Y = []
    while line:
        l = line[:-1].split(',')
        y = [float(l[-1])]
        x = l[:-1]
        x = [float(t) for t in x]
        dev_X.append(x)
        dev_Y.append(y)
        line = f.readline()
        if not line:
            break

saver = tf.train.Saver()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    train_loss_batch = []
    dev_loss = []

    # Fit all trazizning data
    for epoch in range(training_epochs):
        with open('abalone_data_shuffled_train') as f:
            line = f.readline()
            train_X = []
            train_Y = []
            batch_count = 1
            while line:
                while len(train_X) < batch_size:
                    l = line[:-1].split(',')
                    y = [float(l[-1])]
                    x = l[:-1]
                    x = [float(t) for t in x]
                    train_X.append(x)
                    train_Y.append(y)
                    line = f.readline()
                    if not line:
                        break
                # print('training %s batch' % str(batch_count))
                c, _ = sess.run([cost, optimizer], feed_dict={X: np.array(train_X), Y: np.array(train_Y)})
                train_loss_batch.append(c)
                # print ('loss is: %s.' % str(c))
                train_X = []
                train_Y = []
                batch_count += 1
        d_c = sess.run([cost], feed_dict={X: np.array(dev_X), Y: np.array(dev_Y)})

        dev_loss += ([d_c] * (len(train_loss_batch) - len(dev_loss)))
        save_path = saver.save(sess, "/tmp/model.ckpt_" + str(epoch))
        print ('epoch_%s dev error: %s' % (str(epoch), str(d_c)))

    # Graphic display
    assert(len(dev_loss) == len(train_loss_batch))
    plt.plot([i + 1 for i, _ in enumerate(train_loss_batch)], train_loss_batch)
    plt.plot([i + 1 for i, _ in enumerate(dev_loss)], dev_loss)
    plt.legend(['train loss', 'dev loss'])
    plt.xlabel('batch index')
    plt.ylabel('loss')
    plt.show()

with open('abalone_data_shuffled_test') as f:
    line = f.readline()
    test_X = []
    test_Y = []
    while line:
        l = line[:-1].split(',')
        y = [float(l[-1])]
        x = l[:-1]
        x = [float(t) for t in x]
        test_X.append(x)
        test_Y.append(y)
        line = f.readline()
        if not line:
            break

with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt_99")
    predictions, cost = sess.run([a4, cost], feed_dict={X:np.array(test_X), Y:np.array(test_Y)})
    print('Test RMS error: %s' % str(cost))

