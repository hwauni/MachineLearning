# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


# [1, 2, 3, 4, 5, 6, 7, 8, 9]
const1 = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

ti_const1 = tf.tile(const1, [1])
print(ti_const1)
print(tf.shape(ti_const1))
tfutil.print_operation_value(ti_const1)

ti_const1 = tf.tile(const1, [2])
print(ti_const1)
print(tf.shape(ti_const1))
tfutil.print_operation_value(ti_const1)

ti_const1 = tf.tile(const1, [3])
print(ti_const1)
print(tf.shape(ti_const1))
tfutil.print_operation_value(ti_const1)
