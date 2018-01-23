# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [1]
const1 = tf.constant(np.array(x))
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

sn_const1 = tf.shape_n([const1])
print(sn_const1)
tfutil.print_operation_value(sn_const1)

x = [1, 2]
const2 = tf.constant(np.array(x))
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

sn_const2 = tf.shape_n([const2])
print(sn_const2)
tfutil.print_operation_value(sn_const2)

x = [[1, 2], [3, 4]]
const3 = tf.constant(np.array(x))
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

sn_const3 = tf.shape_n([const3])
print(sn_const3)
tfutil.print_operation_value(sn_const3)

x = [[1, 2], [3, 4], [5, 6]]
const4 = tf.constant(np.array(x))
print(const4)
print(tf.shape(const4))
tfutil.print_constant(const4)

sn_const4 = tf.shape_n([const4])
print(sn_const4)
tfutil.print_operation_value(sn_const4)

x = [[[1], [2]], [[3], [4]]]
const5 = tf.constant(np.array(x))
print(const5)
print(tf.shape(const5))
tfutil.print_constant(const5)

sn_const5 = tf.shape_n([const5])
print(sn_const5)
tfutil.print_operation_value(sn_const5)

x = [[[1], [2]], [[3], [4]], [[5], [6]]]
const6 = tf.constant(np.array(x))
print(const6)
print(tf.shape(const6))
tfutil.print_constant(const6)

sn_const6 = tf.shape_n([const6])
print(sn_const6)
tfutil.print_operation_value(sn_const6)
