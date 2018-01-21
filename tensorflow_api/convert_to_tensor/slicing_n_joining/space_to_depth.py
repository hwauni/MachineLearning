# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [[[[1], [2]],
      [[3], [4]]]]

const1 = tf.constant(np.array(x), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

blocksize = 2
std_const1 = tf.space_to_depth(const1, blocksize)
print(std_const1)
print(tf.shape(std_const1))
tfutil.print_operation_value(std_const1)

x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
const2 = tf.constant(np.array(x), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

blocksize = 2
std_const2 = tf.space_to_depth(const2, blocksize)
print(std_const2)
print(tf.shape(std_const2))
tfutil.print_operation_value(std_const2)

x = [[[[1],   [2],  [5],  [6]],
      [[3],   [4],  [7],  [8]],
      [[9],  [10], [13],  [14]],
      [[11], [12], [15],  [16]]]]
const3 = tf.constant(np.array(x), dtype=tf.int32)
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

blocksize = 2
std_const3 = tf.space_to_depth(const3, blocksize)
print(std_const3)
print(tf.shape(std_const3))
tfutil.print_operation_value(std_const3)
