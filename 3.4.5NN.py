import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# use the struct of forward propagation
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# Exploit the 'None' in a dimension of shape can be convenient for
# using different size of batch.when training data,
# data should be split to a small batch,but when testing data,
# all the data can be used at a time.when the data set is small,In
# this way,it can be easy to test,but if the data set is very big,
# it will cause the  memory overflow when we put massive data into a
# batch.
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

# define the processing of  neural network forward propagation
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# define loss function and the algorithm of cross-entropy
y = tf.sigmoid(y)
# tf.reduce_means can be regard as calculate the average to some extent
# tf.clip_by_value means if the value is less than the second parameter
# the value is set to the second parameter,it can ensure the inner of log
# can larger than 0 and it can also avoid the probability larger than 1.
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))
    +(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)

        sess.run(train_step,
                 feed_dict={x: X[start:end],y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                cross_entropy,feed_dict={x: X,y_: Y})
            print("After %d training step(s).cross entropy on all data is %g" %
                (i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
