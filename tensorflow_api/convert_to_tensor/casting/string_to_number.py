# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


var = tf.Variable("1", dtype=tf.string)
tfutil.print_variable(var)
print(var)
num1 = tf.string_to_number(var, out_type=tf.int32)
tfutil.print_operation_value(num1)
print(num1)

const = tf.constant("2", dtype=tf.string)
tfutil.print_constant(const)
print(const)
num2 = tf.string_to_number(const, out_type=tf.float32)
tfutil.print_operation_value(num2)
print(num2)
