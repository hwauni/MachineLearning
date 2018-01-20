# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


# [1, 2, 3, 4, 5, 6, 7, 8, 9]
const1 = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

sp1_const1, sp2_const1 = tf.split(const1, 2, 0)
print(sp1_const1)
print(tf.shape(sp1_const1))
tfutil.print_operation_value(sp1_const1)

print(sp2_const1)
print(tf.shape(sp2_const1))
tfutil.print_operation_value(sp2_const1)

# [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]]
const2 = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

sp1_const2, sp2_const2 = tf.split(const2, 2, 1)
print(sp1_const2)
print(tf.shape(sp1_const2))
tfutil.print_operation_value(sp1_const2)

print(sp2_const2)
print(tf.shape(sp2_const2))
tfutil.print_operation_value(sp2_const2)

# [5, 30]
var1 = tf.Variable(tf.random_normal([5, 30]))
print(var1)
print(tf.shape(var1))

sp1_var1, sp2_var1, sp3_var1 = tf.split(var1, 3, 1)
print(sp1_var1)
print(tf.shape(sp1_var1))

print(sp2_var1)
print(tf.shape(sp2_var1))

print(sp3_var1)
print(tf.shape(sp3_var1))
