# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [0, 1, 2, 3]
const1 = tf.constant(np.array(x)) 
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

mask = np.array([True, False, True, False])
bm_const1 = tf.boolean_mask(const1, mask)
print(bm_const1)
print(tf.shape(bm_const1))
tfutil.print_operation_value(bm_const1)

x = [[1, 2], [3, 4], [5, 6]]
const2 = tf.constant(np.array(x)) 
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

mask = np.array([True, False, True])
bm_const2 = tf.boolean_mask(const2, mask)
print(bm_const2)
print(tf.shape(bm_const2))
tfutil.print_operation_value(bm_const2)

