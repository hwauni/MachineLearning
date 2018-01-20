# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil

'''
n = 10
# images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100 in order
images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]
'''

images = [[[[1], [2]], [[3], [4]]]]

const1 = tf.constant(np.array(images), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

paddings = [[0, 0],[0, 0]]
blocksize = 2
stb_const1 = tf.space_to_batch(const1, paddings, blocksize)
print(stb_const1)
print(tf.shape(stb_const1))
tfutil.print_operation_value(stb_const1)

x = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
const2 = tf.constant(np.array(images), dtype=tf.int32)
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
const3 = tf.constant(np.array(images), dtype=tf.int32)
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
const4 = tf.constant(np.array(images), dtype=tf.int32)
print(const4)
print(tf.shape(const4))
tfutil.print_constant(const4)

paddings = [[0, 0],[0, 0]]
blocksize = 2
stb_const4 = tf.space_to_batch(const4, paddings, blocksize)
print(stb_const4)
print(tf.shape(stb_const4))
tfutil.print_operation_value(stb_const4)
