# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


const1 = tf.constant(1, dtype=tf.float32)
tfutil.print_constant(const1)
print(const1)
int32_1 = tf.to_int32(const1)
tfutil.print_operation_value(int32_1)
print(int32_1)

const2 = tf.constant([2, 3], dtype=tf.float32)
tfutil.print_constant(const2)
print(const2)
int32_2 = tf.to_int32(const2)
tfutil.print_operation_value(int32_2)
print(int32_2)

var1 = tf.Variable(4, dtype=tf.float32)
tfutil.print_variable(var1)
print(var1)
int32_3 = tf.to_int32(var1)
tfutil.print_operation_value(int32_3)
print(int32_3)

var2 = tf.Variable([5, 6], dtype=tf.float32)
tfutil.print_variable(var2)
print(var2)
int32_4 = tf.to_int32(var2)
tfutil.print_operation_value(int32_4)
print(int32_4)
