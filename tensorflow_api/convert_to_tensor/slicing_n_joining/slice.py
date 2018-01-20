# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


# [1, 2, 3, 4, 5, 6, 7, 8, 9]
const1 = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)
sl_const1 = tf.slice(const1, [2], [3])
print(sl_const1)
print(tf.shape(sl_const1))
tfutil.print_operation_value(sl_const1)


# [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19,20]]
const2 = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19,20]])
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

sl_const2 = tf.slice(const2, [0, 1], [1, 3])
print(sl_const2)
print(tf.shape(sl_const2))
tfutil.print_operation_value(sl_const2)

sl_const2 = tf.slice(const2, [0, 2], [1, 3])
print(sl_const2)
print(tf.shape(sl_const2))
tfutil.print_operation_value(sl_const2)

sl_const2 = tf.slice(const2, [1, 1], [1, 3])
print(sl_const2)
print(tf.shape(sl_const2))
tfutil.print_operation_value(sl_const2)

sl_const2 = tf.slice(const2, [1, 2], [1, 3])
print(sl_const2)
print(tf.shape(sl_const2))
tfutil.print_operation_value(sl_const2)

# [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]]
const3 = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [1, 1, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [1, 1, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 1, 0], [1, 1, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 1, 0], [1, 1, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [1, 0, 0], [1, 1, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [1, 0, 0], [1, 1, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [1, 1, 0], [1, 1, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [1, 1, 0], [1, 1, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [2, 0, 0], [1, 1, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [2, 0, 0], [1, 1, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [2, 1, 0], [1, 1, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [2, 1, 0], [1, 1, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

# [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]]
sl_const3 = tf.slice(const3, [0, 0, 0], [1, 1, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [1, 1, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [1, 1, 3])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [1, 2, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [1, 2, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [1, 2, 3])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [3, 1, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [3, 2, 1])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [3, 1, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [3, 2, 2])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [3, 1, 3])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)

sl_const3 = tf.slice(const3, [0, 0, 0], [3, 2, 3])
print(sl_const3)
print(tf.shape(sl_const3))
tfutil.print_operation_value(sl_const3)
