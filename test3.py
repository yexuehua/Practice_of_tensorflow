import tensorflow as tf

input1 = tf.constant(2.0)
input2 = tf.constant(3.0)
input3 = tf.constant(5.0)

intermd = tf.add(input1,input2)
mul = tf.multiply(input3,intermd)

with tf.Session() as sess:
    result = sess.run([mul,intermd])
    print(result)