# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

# [1, 2, 1, 3, 1, 1]
const1 = tf.constant([[ [[ [[1]], [[2]], [[3]] ]], [[ [[4]], [[5]], [[6]] ]] ]])

print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)
sq_const1 = tf.squeeze(const1)
print(sq_const1)
print(tf.shape(sq_const1))
tfutil.print_operation_value(sq_const1)


# [1, 2, 1, 3, 1, 1]
const2 = tf.constant([[ [[ [[1]], [[2]], [[3]] ]], [[ [[4]], [[5]], [[6]] ]] ]])
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

sq_const2 = tf.squeeze(const2, [0])
print(sq_const2)
print(tf.shape(sq_const2))
tfutil.print_operation_value(sq_const2)

sq_const2 = tf.squeeze(const2, [2])
print(sq_const2)
print(tf.shape(sq_const2))
tfutil.print_operation_value(sq_const2)

sq_const2 = tf.squeeze(const2, [0, 2])
print(sq_const2)
print(tf.shape(sq_const2))
tfutil.print_operation_value(sq_const2)

sq_const2 = tf.squeeze(const2, [2, 4])
print(sq_const2)
print(tf.shape(sq_const2))
tfutil.print_operation_value(sq_const2)

sq_const2 = tf.squeeze(const2, [0, 2, 4])
print(sq_const2)
print(tf.shape(sq_const2))
tfutil.print_operation_value(sq_const2)

sq_const2 = tf.squeeze(const2, [0, 2, 4, 5])
print(sq_const2)
print(tf.shape(sq_const2))
tfutil.print_operation_value(sq_const2)
