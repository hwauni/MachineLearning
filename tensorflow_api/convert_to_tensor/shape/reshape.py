# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


const1 = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(const1)
tfutil.print_constant(const1)
rs_const1 = tf.reshape(const1, [3, 3])
print(rs_const1)
tfutil.print_operation_value(rs_const1)

const2 = tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
print(const2)
tfutil.print_constant(const2)
rs_const2 = tf.reshape(const2, [2, 4])
print(rs_const2)
tfutil.print_operation_value(rs_const2)

const3 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
print(const3)
tfutil.print_constant(const3)
rs_const3 = tf.reshape(const3, [-1])
print(rs_const3)
tfutil.print_operation_value(rs_const3)

const4 = tf.constant([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])
print(const4)
tfutil.print_constant(const4)
rs_const4 = tf.reshape(const4, [2, -1])
print(rs_const4)
tfutil.print_operation_value(rs_const4)

const5 = tf.constant([[1, 1, 1, 2, 2, 2, 3, 3, 3], [4, 4, 4, 5, 5, 5, 6, 6, 6]])
print(const5)
tfutil.print_constant(const5)
rs_const5 = tf.reshape(const5, [2, -1, 3])
print(rs_const5)
tfutil.print_operation_value(rs_const5)

const6 = tf.constant([7])
print(const6)
tfutil.print_constant(const6)
rs_const6 = tf.reshape(const6, [])
print(rs_const6)
tfutil.print_operation_value(rs_const6)

var1 = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(var1)
tfutil.print_variable(var1)
rs_var1 = tf.reshape(var1, [3, 3])
print(rs_var1)
tfutil.print_operation_value(rs_var1)

var2 = tf.Variable([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
print(var2)
tfutil.print_variable(var2)
rs_var2 = tf.reshape(var2, [2, 4])
print(rs_var2)
tfutil.print_operation_value(rs_var2)

var3 = tf.Variable([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
print(var3)
tfutil.print_variable(var3)
rs_var3 = tf.reshape(var3, [-1])
print(rs_var3)
tfutil.print_operation_value(rs_var3)

var4 = tf.Variable([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])
print(var4)
tfutil.print_variable(var4)
rs_var4 = tf.reshape(var4, [2, -1])
print(rs_var4)
tfutil.print_operation_value(rs_var4)

var5 = tf.Variable([[1, 1, 1, 2, 2, 2, 3, 3, 3], [4, 4, 4, 5, 5, 5, 6, 6, 6]])
print(var5)
tfutil.print_variable(var5)
rs_var5 = tf.reshape(var5, [2, -1, 3])
print(rs_var5)
tfutil.print_operation_value(rs_var5)

var6 = tf.Variable([7])
print(var6)
tfutil.print_variable(var6)
rs_var6 = tf.reshape(var6, [])
print(rs_var6)
tfutil.print_operation_value(rs_var6)

