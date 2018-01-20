# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil

x = [[1, 2, 3, 4, 0, 0, 0],
     [1, 2, 3, 0, 0, 0, 0],
     [1, 2, 3, 4, 5, 6, 7]]
const1 = tf.constant(np.array(x), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

seq_lens = [4, 3, 7]
rs_const1 = tf.reverse_sequence(const1, seq_lens, seq_dim=1, batch_dim=0)
print(rs_const1)
print(tf.shape(rs_const1))
tfutil.print_operation_value(rs_const1)

x = [[[1, 2, 3, 4, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0]],
     [[1, 2, 3, 4, 5, 0, 0], [0, 2, 1, 4, 0, 0, 0]],
     [[1, 2, 3, 4, 5, 6 ,7], [1, 2, 3, 4, 0, 6, 0]]]
const2 = tf.constant(np.array(x), dtype=tf.int32)
print(const2)
print(tf.shape(const2))
tfutil.print_constant(const2)

seq_lens = [4, 3, 5]
rs_const2 = tf.reverse_sequence(const2, seq_lens, seq_dim=2, batch_dim=0)
print(rs_const2)
print(tf.shape(rs_const2))
tfutil.print_operation_value(rs_const2)

seq_lens = [4, 3]
rs_const2 = tf.reverse_sequence(const2, seq_lens, seq_dim=2, batch_dim=1)
print(rs_const2)
print(tf.shape(rs_const2))
tfutil.print_operation_value(rs_const2)
