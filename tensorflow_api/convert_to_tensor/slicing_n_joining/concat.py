# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


# [[1, 2, 3], [4, 5, 6]]
# [[7, 8, 9], [10, 11, 12]]
const1 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)
const2 = tf.constant(np.array([[7, 8, 9], [10, 11, 12]]), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

cc_const1 = tf.concat([const1, const2], 0)
print(cc_const1)
print(tf.shape(cc_const1))
tfutil.print_operation_value(cc_const1)

cc_const1 = tf.concat([const1, const2], 1)
print(cc_const1)
print(tf.shape(cc_const1))
tfutil.print_operation_value(cc_const1)
