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

x = [[1, 2], [3, 4]]
const1 = tf.constant(np.array(x)) 
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

indice = [[0,1], [2, 3]]
dys_const1 = tf.dynamic_stitch(indice, const1)
print(dys_const1[0])
print(tf.shape(dys_const1[0]))
tfutil.print_operation_value(dys_const1[0])




x = [[1, 2], [3, 4]]
const1 = tf.constant(np.array(x)) 
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

y = [1, 1]
row_to_add = tf.constant(np.array(y))
print(row_to_add)
print(tf.shape(row_to_add))
tfutil.print_constant(row_to_add)

original_row = const1[0]
print(original_row)
print(tf.shape(original_row))
tfutil.print_constant(original_row)

updated_row = original_row + row_to_add
print(updated_row)
print(tf.shape(updated_row))
tfutil.print_operation_value(updated_row)

unchanged_indices = tf.range(tf.size(const1)) 
print(unchanged_indices)
print(tf.shape(unchanged_indices))
tfutil.print_operation_value(unchanged_indices)

changed_indices = tf.range(const1.get_shape()[0]) 
print(changed_indices)
print(tf.shape(changed_indices))
tfutil.print_operation_value(changed_indices)

a_flat = tf.reshape(const1, [-1]) 
print(a_flat)
print(tf.shape(a_flat))
tfutil.print_operation_value(a_flat)

updated_a_flat = tf.dynamic_stitch([unchanged_indices, changed_indices], [a_flat, updated_row]) 
print(updated_a_flat)
print(tf.shape(updated_a_flat))
tfutil.print_operation_value(updated_a_flat)

updated_a = tf.reshape(updated_a_flat, const1.get_shape()) 
print(updated_a)
print(tf.shape(updated_a))
tfutil.print_operation_value(updated_a) 







# Scalar partitions.
indice = [[0,1], [2, 3]]
dys_const1 = tf.dynamic_stitch(indice, const1)
print(dys_const1[0])
print(tf.shape(dys_const1[0]))
tfutil.print_operation_value(dys_const1[0])

print(dyp_const1[1])
print(tf.shape(dyp_const1[1]))
tfutil.print_operation_value(dyp_const1[1])

# Apply function (increments x_i) on elements for which a certain condition
# apply (x_i != -1 in this example).
x = [0.1, -1., 5.2, 4.3, -1., 7.4]
const1 = tf.constant(np.array(x))
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

condition_mask=tf.not_equal(x,tf.constant(-1.))
print(condition_mask)
print(tf.shape(condition_mask))
tfutil.print_constant(condition_mask)

partitioned_data = tf.dynamic_partition(x, 
    tf.cast(condition_mask, tf.int32) , 2)
print(partitioned_data)
print(tf.shape(partitioned_data))
tfutil.print_constant(partitioned_data)

partitioned_data[1] = partitioned_data[1] + 1.0
print(partitioned_data[1])
print(tf.shape(partitioned_data[1]))
tfutil.print_constant(partitioned_data[1])

condition_indices = tf.dynamic_partition(
    tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
print(condition_indices)
print(tf.shape(condition_indices))
tfutil.print_constant(condition_indices)

ds_const1 = tf.dynamic_stitch(condition_indices, partitioned_data)
print(ds_const1)
print(tf.shape(ds_const1))
tfutil.print_constant(ds_const1)

# Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
# unchanged.
