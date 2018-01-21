# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]

const1 = tf.constant(np.array(x), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

# crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
# [batch, height, width, depth]
# height = height_pad - crop_top - crop_bottom
# width = width_pad - crop_left - crop_right

corps = [[0, 0],[0, 1]]
blocksize = 1
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

corps = [[0, 0],[0, 1]]
blocksize = 1
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

corps = [[0, 1],[0, 0]]
blocksize = 1
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

corps = [[1, 0],[0, 0]]
blocksize = 1
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

corps = [[0, 0],[0, 0]]
blocksize = 2
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

corps = [[0, 0],[1, 1]]
blocksize = 2
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

corps = [[1, 1],[0, 0]]
blocksize = 2
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

corps = [[1, 1],[1, 1]]
blocksize = 2
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

x = [[[[ 1,  2,  3]]],
 [[[ 4,  5,  6]]],
 [[[ 7,  8,  9]]],
 [[[10, 11, 12]]]]
const2 = tf.constant(np.array(x), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

corps = [[0, 0],[0, 0]]
blocksize = 2
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

x = [[[[1], [3]], [[9], [11]]],
     [[[2], [4]], [[10], [12]]],
     [[[5], [7]], [[13], [15]]],
     [[[6], [8]], [[14], [16]]]]
const3 = tf.constant(np.array(x), dtype=tf.int32)
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

corps = [[0, 0],[0, 0]]
blocksize = 2
bts_const3 = tf.batch_to_space(const3, corps, blocksize)
print(bts_const3)
print(tf.shape(bts_const3))
tfutil.print_operation_value(bts_const3)

x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
     [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
const4 = tf.constant(np.array(x), dtype=tf.int32)
print(const4)
print(tf.shape(const4))
tfutil.print_constant(const4)

corps = [[0, 0],[0, 0]]
blocksize = 2
bts_const4 = tf.batch_to_space(const4, corps, blocksize)
print(bts_const4)
print(tf.shape(bts_const4))
tfutil.print_operation_value(bts_const4)
