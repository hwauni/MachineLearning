# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


const1 = tf.constant([1, 2, 3])
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)
ed_const1 = tf.expand_dims(const1, 0)
print(ed_const1)
tfutil.print_operation_value(ed_const1)
print(tf.shape(ed_const1))

ed_const1 = tf.expand_dims(const1, 1)
print(ed_const1)
tfutil.print_operation_value(ed_const1)
print(tf.shape(ed_const1))

ed_const1 = tf.expand_dims(const1, -1)
print(ed_const1)
tfutil.print_operation_value(ed_const1)
print(tf.shape(ed_const1))


const2 = tf.constant([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)
ed_const2 = tf.expand_dims(const2, 0)
print(ed_const2)
tfutil.print_operation_value(ed_const2)
print(tf.shape(ed_const2))

ed_const2 = tf.expand_dims(const2, 1)
print(ed_const2)
tfutil.print_operation_value(ed_const2)
print(tf.shape(ed_const2))

ed_const2 = tf.expand_dims(const2, 2)
print(ed_const2)
tfutil.print_operation_value(ed_const2)
print(tf.shape(ed_const2))

ed_const2 = tf.expand_dims(const2, 3)
print(ed_const2)
tfutil.print_operation_value(ed_const2)
print(tf.shape(ed_const2))

ed_const2 = tf.expand_dims(const2, -1)
print(ed_const2)
tfutil.print_operation_value(ed_const2)
print(tf.shape(ed_const2))


