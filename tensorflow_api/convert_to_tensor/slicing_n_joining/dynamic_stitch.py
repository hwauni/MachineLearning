# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


'''
# merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
# Scalar indices:
  merged[indices[m], ...] = data[m][...]
# Vector indices:
  merged[indices[m][i], ...] = data[m][i, ...]
merged.shape = [max(indices)] + constant
'''

x = [10, 20] 
const1 = tf.constant(np.array(x))
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

# Scalar partitions.
partitions = 1
num_partitions = 2
dyp_const1 = tf.dynamic_partition(const1, partitions, num_partitions)
print(dyp_const1[0])
print(tf.shape(dyp_const1[0]))
tfutil.print_operation_value(dyp_const1[0])

print(dyp_const1[1])
print(tf.shape(dyp_const1[1]))
tfutil.print_operation_value(dyp_const1[1])

print(dyp_const1)
print(tf.shape(dyp_const1))
tfutil.print_operation_value(dyp_const1)
