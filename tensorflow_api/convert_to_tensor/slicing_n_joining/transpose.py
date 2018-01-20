# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil

'''
A=[2,3,4] matrix, using perm(1,0,2) will get B=[3,2,4]

Index = (0,1,2)
A     = [2,3,4]
Perm  = (1,0,2)
B     = (3,2,4)  --> Perm 1 from Index 1 (3), Perm 0 from Index 0 (2), Perm 2 from Index 2 (4) --> so get (3,2,4
'''

x = [[1, 2, 3, 4, 5, 6]]
const1 = tf.constant(np.array(x), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

perm = [1, 0]
trans_const1 = tf.transpose(const1, perm)
print(trans_const1)
print(tf.shape(trans_const1))
tfutil.print_operation_value(trans_const1)

perm = [0, 1]
trans_const1 = tf.transpose(const1, perm)
print(trans_const1)
print(tf.shape(trans_const1))
tfutil.print_operation_value(trans_const1)


x = [[1, 2, 3], [4, 5, 6]]
const2 = tf.constant(np.array(x), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

perm = [1, 0] 
trans_const2 = tf.transpose(const2, perm)
print(trans_const2)
print(tf.shape(trans_const2))
tfutil.print_operation_value(trans_const2)

perm = [0, 1] 
trans_const2 = tf.transpose(const2, perm)
print(trans_const2)
print(tf.shape(trans_const2))
tfutil.print_operation_value(trans_const2)

x = [[[ 1,  2,  3], 
      [ 4,  5,  6]],
     [[ 7,  8,  9],
      [10, 11, 12]]]
const3 = tf.constant(np.array(x), dtype=tf.int32)
print(const3)
print(tf.shape(const3))
tfutil.print_constant(const3)

perm = [1, 0, 2] 
trans_const3 = tf.transpose(const3, perm)
print(trans_const3)
print(tf.shape(trans_const3))
tfutil.print_operation_value(trans_const3)

perm = [2, 0, 1] 
trans_const3 = tf.transpose(const3, perm)
print(trans_const3)
print(tf.shape(trans_const3))
tfutil.print_operation_value(trans_const3)

perm = [2, 1, 0] 
trans_const3 = tf.transpose(const3, perm)
print(trans_const3)
print(tf.shape(trans_const3))
tfutil.print_operation_value(trans_const3)

perm = [1, 2, 0] 
trans_const3 = tf.transpose(const3, perm)
print(trans_const3)
print(tf.shape(trans_const3))
tfutil.print_operation_value(trans_const3)

perm = [0, 2, 1] 
trans_const3 = tf.transpose(const3, perm)
print(trans_const3)
print(tf.shape(trans_const3))
tfutil.print_operation_value(trans_const3)

perm = [0, 1, 2] 
trans_const3 = tf.transpose(const3, perm)
print(trans_const3)
print(tf.shape(trans_const3))
tfutil.print_operation_value(trans_const3)
