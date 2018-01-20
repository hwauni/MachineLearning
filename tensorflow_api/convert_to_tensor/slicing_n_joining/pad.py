# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


# [[1, 2, 3], [4, 5, 6]]
const1 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

paddings =  [[0, 0,], [0, 1]]
pad_const1 = tf.pad(const1, paddings, "CONSTANT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 0,], [1, 0]]
pad_const1 = tf.pad(const1, paddings, "CONSTANT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 0,], [1, 1]]
pad_const1 = tf.pad(const1, paddings, "CONSTANT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 1,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "CONSTANT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 0,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "CONSTANT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "CONSTANT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [1, 1]]
pad_const1 = tf.pad(const1, paddings, "CONSTANT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [2, 2]]
pad_const1 = tf.pad(const1, paddings, "CONSTANT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 0,], [0, 1]]
pad_const1 = tf.pad(const1, paddings, "REFLECT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 0,], [1, 0]]
pad_const1 = tf.pad(const1, paddings, "REFLECT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 0,], [1, 1]]
pad_const1 = tf.pad(const1, paddings, "REFLECT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 1,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "REFLECT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 0,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "REFLECT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "REFLECT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [1, 1]]
pad_const1 = tf.pad(const1, paddings, "REFLECT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [2, 2]]
pad_const1 = tf.pad(const1, paddings, "REFLECT")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)


paddings =  [[0, 0,], [0, 1]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 0,], [1, 0]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 0,], [1, 1]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[0, 1,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 0,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [0, 0]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [1, 1]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [2, 2]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)

paddings =  [[1, 1,], [3, 3]]
pad_const1 = tf.pad(const1, paddings, "SYMMETRIC")
print(pad_const1)
print(tf.shape(pad_const1))
tfutil.print_operation_value(pad_const1)
