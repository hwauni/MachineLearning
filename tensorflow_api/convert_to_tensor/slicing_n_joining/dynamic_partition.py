# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil


# outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
# outputs[i] = pack([data[js, ...] for js if partitions[js] == i])

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


x = [10, 20, 30, 40, 50]
const2 = tf.constant(np.array(x))
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

# Vector partitions.
partitions = [0, 0, 1, 1, 0]
num_partitions = 2
dyp_const2 = tf.dynamic_partition(const2, partitions, num_partitions)
print(dyp_const2[0])
print(tf.shape(dyp_const2[0]))
tfutil.print_operation_value(dyp_const2[0])

print(dyp_const2[1])
print(tf.shape(dyp_const2[1]))
tfutil.print_operation_value(dyp_const2[1])

print(dyp_const2)
print(tf.shape(dyp_const2))
tfutil.print_operation_value(dyp_const2)
