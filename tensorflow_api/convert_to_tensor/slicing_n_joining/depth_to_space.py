# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [[[[1, 2, 3, 4]]]]

const1 = tf.constant(np.array(x), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

blocksize = 2
dts_const1 = tf.depth_to_space(const1, blocksize)
print(dts_const1)
print(tf.shape(dts_const1))
tfutil.print_operation_value(dts_const1)

x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
const2 = tf.constant(np.array(x), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

blocksize = 2
dts_const2 = tf.depth_to_space(const2, blocksize)
print(dts_const2)
print(tf.shape(dts_const2))
tfutil.print_operation_value(dts_const2)

x =  [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
const3 = tf.constant(np.array(x), dtype=tf.int32)
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

blocksize = 2
dts_const3 = tf.depth_to_space(const3, blocksize)
print(dts_const3)
print(tf.shape(dts_const3))
tfutil.print_operation_value(dts_const3)
