# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [[[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
        [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]]]

const1 = tf.constant(np.array(x), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

dims = [3] 
dims_const1 = tf.reverse(const1, dims)
print(dims_const1)
print(tf.shape(dims_const1))
tfutil.print_operation_value(dims_const1)

dims = [1] 
dims_const1 = tf.reverse(const1, dims)
print(dims_const1)
print(tf.shape(dims_const1))
tfutil.print_operation_value(dims_const1)

dims = [2] 
dims_const1 = tf.reverse(const1, dims)
print(dims_const1)
print(tf.shape(dims_const1))
tfutil.print_operation_value(dims_const1)
