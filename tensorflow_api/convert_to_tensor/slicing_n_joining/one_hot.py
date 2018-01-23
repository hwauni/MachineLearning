# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [0, 1, 2]
const1 = tf.constant(np.array(x)) 
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

depth = 3
oh_const1 = tf.one_hot(const1, depth)
print(oh_const1)
print(tf.shape(oh_const1))
tfutil.print_operation_value(oh_const1)

x = [0, 2, -1, 1]
const2 = tf.constant(np.array(x)) 
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

depth = 3
bm_const2 = tf.one_hot(const2, depth, 
    on_value=5.0, off_value=0.0, axis=-1)
print(bm_const2)
print(tf.shape(bm_const2))
tfutil.print_operation_value(bm_const2)

x = [[0, 2], [1, -1]]
const3 = tf.constant(np.array(x)) 
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

depth = 3
bm_const3 = tf.one_hot(const3, depth, 
    on_value=1.0, off_value=0.0, axis=-1)
print(bm_const3)
print(tf.shape(bm_const3))
tfutil.print_operation_value(bm_const3)
