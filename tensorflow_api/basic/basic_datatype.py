# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf

ph = tf.placeholder(tf.float32, shape=[3, 1])
var = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)
const = tf.constant([10, 20, 30, 40, 50], dtype=tf.float32)

print(ph)
print(var)
print(const)

# constant section
sess = tf.Session()
result = sess.run(const)
print(result)
sess.close()

a = tf.constant([5])
b = tf.constant([10])
c = tf.constant([2])
d = a * b + c
sess = tf.Session()
result = sess.run(d)
print(result)
sess.close()

# variable section
var1 = tf.Variable([5])
var2 = tf.Variable([3])
var3 = tf.Variable([2])
var4 = var1 * var2 + var3

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
result = sess.run(var4)
print(result)
sess.close()

# placeholder section
value1 = 5
value2 = 3
value3 = 2
ph1 = tf.placeholder(tf.float32)
ph2 = tf.placeholder(tf.float32)
ph3 = tf.placeholder(tf.float32)
ph4 = ph1 * ph2 + ph3

result_value = ph1 * ph2 + ph3
feed_dict = {ph1: value1, ph2: value2, ph3: value3}
sess = tf.Session()
result = sess.run(result_value, feed_dict=feed_dict)
print(result)
sess.close()
