# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


x = [[1, 2, 3],
     [4, 5, 6],  
     [7, 8, 9]] 
const1 = tf.constant(np.array(x), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

idx_const2 = tf.constant([1, 0, 2])
print(idx_const2)
print(tf.shape(idx_const2))
tfutil.print_constant(idx_const2)

idx_flattened = tf.range(0, const1.shape[0]) * const1.shape[1] + idx_const2
print(idx_flattened)
print(tf.shape(idx_flattened))
tfutil.print_constant(idx_flattened)

# partial code 1
print(const1.shape[0])
tmp1 = tf.range(0, const1.shape[0])
print(tmp1)
print(tf.shape(tmp1))
tfutil.print_constant(tmp1)

# partial code 2
print(const1.shape[1])
tmp2 = tf.range(0, const1.shape[0]) * const1.shape[1]
print(tmp2)
print(tf.shape(tmp2))
tfutil.print_constant(tmp2)

# partial code 3
tmp3 = tmp2 + idx_const2
print(tmp3)
print(tf.shape(tmp3))
tfutil.print_constant(tmp3)

params = tf.reshape(x, [-1])  
print(params)
print(tf.shape(params))
tfutil.print_constant(params)

gather_const1 = tf.gather(params,         # flatten input
              idx_flattened)  # use flattened indices
print(gather_const1)
print(tf.shape(gather_const1))
tfutil.print_constant(gather_const1)
