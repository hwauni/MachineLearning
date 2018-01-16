# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


const1 = tf.constant(1)
tfutil.print_constant(const1)
print(const1)
tfutil.print_operation_value(tf.rank(const1))

const2 = tf.constant([1, 2])
tfutil.print_constant(const2)
print(const2)
tfutil.print_operation_value(tf.rank(const2))

const3 = tf.constant([[1, 2], [3, 4]])
tfutil.print_constant(const3)
print(const3)
tfutil.print_operation_value(tf.rank(const3))

const4 = tf.constant([[1, 2], [3, 4], [5, 6]])
tfutil.print_constant(const4)
print(const4)
tfutil.print_operation_value(tf.rank(const4))

const5 = tf.constant([[[1], [2]], [[3], [4]]])
tfutil.print_constant(const5)
print(const5)
tfutil.print_operation_value(tf.rank(const5))

const6 = tf.constant([[[1], [2]], [[3], [4]], [[5], [6]]])
tfutil.print_constant(const6)
print(const6)
tfutil.print_operation_value(tf.rank(const6))

var1 = tf.Variable(1)
tfutil.print_variable(var1)
print(var1)
tfutil.print_operation_value(tf.rank(var1))

var2 = tf.Variable([1, 2])
tfutil.print_variable(var2)
print(var2)
tfutil.print_operation_value(tf.rank(var2))

var3 = tf.Variable([[1, 2], [3, 4]])
tfutil.print_variable(var3)
print(var3)
tfutil.print_operation_value(tf.rank(var3))

var4 = tf.Variable([[1, 2], [3, 4], [5, 6]])
tfutil.print_variable(var4)
print(var4)
tfutil.print_operation_value(tf.rank(var4))

var5 = tf.Variable([[[1], [2]], [[3], [4]]])
tfutil.print_variable(var5)
print(var5)
tfutil.print_operation_value(tf.rank(var5))

var6 = tf.Variable([[[1], [2]], [[3], [4]], [[5], [6]]])
tfutil.print_variable(var6)
print(var6)
tfutil.print_operation_value(tf.rank(var6))
