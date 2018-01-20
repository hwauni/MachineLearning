# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil

const1 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

const2 = tf.constant(np.array([4, 5, 6]), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

const3 = tf.constant(np.array([7, 8, 9]), dtype=tf.int32)
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

pack_const1 = tf.stack([const1, const2, const3])
print(pack_const1)
print(tf.shape(pack_const1))
tfutil.print_operation_value(pack_const1)

up_const1, up_const2, up_const3 = tf.unstack(pack_const1)
print(up_const1)
print(tf.shape(up_const1))
tfutil.print_constant(up_const1)

print(up_const2)
print(tf.shape(up_const2))
tfutil.print_constant(up_const2)

print(up_const3)
print(tf.shape(up_const3))
tfutil.print_constant(up_const3)

pack_const1 = tf.stack([const1, const2, const3], axis=1)
print(pack_const1)
print(tf.shape(pack_const1))
tfutil.print_operation_value(pack_const1)

up_const1, up_const2, up_const3 = tf.unstack(pack_const1)
print(up_const1)
print(tf.shape(up_const1))
tfutil.print_constant(up_const1)

print(up_const2)
print(tf.shape(up_const2))
tfutil.print_constant(up_const2)

print(up_const3)
print(tf.shape(up_const3))
tfutil.print_constant(up_const3)
