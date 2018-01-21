# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


# [d_0, ..., d_{Q-2}, params.shape[K], ..., params.shape[P-1]].
x = [['a', 'b'], ['c', 'd']]
const1 = tf.constant(np.array(x))
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

# Simple indexing into a matrix:
indices = [[0, 0], [1, 1]]
gn_const1 = tf.gather_nd(const1, indices)
print(gn_const1)
print(tf.shape(gn_const1))
tfutil.print_operation_value(gn_const1)

# Slice indexing into a matrix:
indices = [[1], [0]]
gn_const1 = tf.gather_nd(const1, indices)
print(gn_const1)
print(tf.shape(gn_const1))
tfutil.print_operation_value(gn_const1)

# Batched indexing into a matrix:
indices = [[[0, 0]], [[0, 1]]]
gn_const1 = tf.gather_nd(const1, indices)
print(gn_const1)
print(tf.shape(gn_const1))
tfutil.print_operation_value(gn_const1)

# Batched slice indexing into a matrix:
indices = [[[1]], [[0]]]
gn_const1 = tf.gather_nd(const1, indices)
print(gn_const1)
print(tf.shape(gn_const1))
tfutil.print_operation_value(gn_const1)


x = [[['a0', 'b0'], ['c0', 'd0']],
    [['a1', 'b1'], ['c1', 'd1']]]
const2 = tf.constant(np.array(x))
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

# Indexing into a 3-tensor:
indices = [[1]]
gn_const2 = tf.gather_nd(const2, indices)
print(gn_const2)
print(tf.shape(gn_const2))
tfutil.print_operation_value(gn_const2)

indices = [[0, 1], [1, 0]]
gn_const2 = tf.gather_nd(const2, indices)
print(gn_const2)
print(tf.shape(gn_const2))
tfutil.print_operation_value(gn_const2)

indices = [[0, 0, 1], [1, 0, 1]]
gn_const2 = tf.gather_nd(const2, indices)
print(gn_const2)
print(tf.shape(gn_const2))
tfutil.print_operation_value(gn_const2)

# Batched indexing into a 3-tensor:
indices = [[[1]], [[0]]]
gn_const2 = tf.gather_nd(const2, indices)
print(gn_const2)
print(tf.shape(gn_const2))
tfutil.print_operation_value(gn_const2)

indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
gn_const2 = tf.gather_nd(const2, indices)
print(gn_const2)
print(tf.shape(gn_const2))
tfutil.print_operation_value(gn_const2)

indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
gn_const2 = tf.gather_nd(const2, indices)
print(gn_const2)
print(tf.shape(gn_const2))
tfutil.print_operation_value(gn_const2)
