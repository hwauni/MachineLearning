# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [[[[1], [2]], [[3], [4]]]]

const1 = tf.constant(np.array(x), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

# return: [batch*block_size*block_size, height_pad/block_size, width_pad/block_size, depth]
# paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
# height_pad = pad_top + height + pad_bottom
# width_pad = pad_left + width + pad_right
# blocksize = Both height_pad and width_pad must be divisible by block_size.
paddings = [[0, 0],[0, 0]]
blocksize = 1
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

paddings = [[0, 0],[0, 1]]
blocksize = 1
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

paddings = [[0, 0],[1, 1]]
blocksize = 1
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

paddings = [[0, 1],[1, 1]]
blocksize = 1
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

paddings = [[1, 1],[1, 1]]
blocksize = 1
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

paddings = [[1, 1],[1, 1]]
blocksize = 2
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

paddings = [[0, 0],[0, 0]]
blocksize = 2
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

paddings = [[0, 0],[1, 1]]
blocksize = 2
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

paddings = [[1, 1],[1, 1]]
blocksize = 2
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

x = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
const2 = tf.constant(np.array(x), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

paddings = [[0, 0],[0, 0]]
blocksize = 2
stb_const2 = tf.space_to_batch(const2, paddings, blocksize)
print(stb_const2)
print(tf.shape(stb_const2))
tfutil.print_operation_value(stb_const2)

x = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
const3 = tf.constant(np.array(x), dtype=tf.int32)
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

paddings = [[0, 0],[0, 0]]
blocksize = 2
stb_const3 = tf.space_to_batch(const3, paddings, blocksize)
print(stb_const3)
print(tf.shape(stb_const3))
tfutil.print_operation_value(stb_const3)

x = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]]], [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
const4 = tf.constant(np.array(x), dtype=tf.int32)
print(const4)
print(tf.shape(const4))
tfutil.print_constant(const4)

paddings = [[0, 0],[0, 0]]
blocksize = 2
stb_const4 = tf.space_to_batch(const4, paddings, blocksize)
print(stb_const4)
print(tf.shape(stb_const4))
tfutil.print_operation_value(stb_const4)
