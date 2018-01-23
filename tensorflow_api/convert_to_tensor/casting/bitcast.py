# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = 37.0
const1 = tf.constant(x) 
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

bc_const1 = tf.bitcast(const1, tf.int32)
print(bc_const1)
print(tf.shape(bc_const1))
tfutil.print_operation_value(bc_const1)

x = -1
invert_bits = tf.constant(x) - bc_const1
print(invert_bits)
print(tf.shape(invert_bits))
tfutil.print_operation_value(invert_bits)

bc_to_float = tf.bitcast(invert_bits, tf.float32)
print(bc_to_float)
print(tf.shape(bc_to_float))
tfutil.print_operation_value(bc_to_float)
