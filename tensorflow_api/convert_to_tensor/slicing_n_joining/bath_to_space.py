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

images = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]

const1 = tf.constant(np.array(images), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

corps = [[0, 0],[0, 0]]
blocksize = 2
bts_const1 = tf.batch_to_space(const1, corps, blocksize)
print(bts_const1)
print(tf.shape(bts_const1))
tfutil.print_operation_value(bts_const1)

x = [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
const2 = tf.constant(np.array(images), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

corps = [[0, 0],[0, 0]]
blocksize = 2
bts_const2 = tf.batch_to_space(const2, corps, blocksize)
print(bts_const2)
print(tf.shape(bts_const2))
tfutil.print_operation_value(bts_const2)

x = [[[[1], [3]], [[9], [11]]],
     [[[2], [4]], [[10], [12]]],
     [[[5], [7]], [[13], [15]]],
     [[[6], [8]], [[14], [16]]]]
const3 = tf.constant(np.array(images), dtype=tf.int32)
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
const4 = tf.constant(np.array(images), dtype=tf.int32)
print(const4)
print(tf.shape(const4))
tfutil.print_constant(const4)

corps = [[0, 0],[0, 0]]
blocksize = 2
bts_const4 = tf.batch_to_space(const4, corps, blocksize)
print(bts_const4)
print(tf.shape(bts_const4))
tfutil.print_operation_value(bts_const4)
