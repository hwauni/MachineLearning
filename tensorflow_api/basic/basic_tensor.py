# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

# constant
const_s_0 = tf.constant(1)
const_v_1_1 = tf.constant([1, 2])
const_v_1_2 = tf.constant([1, 2, 3])
const_m_2_1 = tf.constant([[1, 2, 3], [4, 5, 6]])
const_m_2_2 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
const_n_3_1 = tf.constant([[[2], [4]], [[8], [10]], [[14], [16]]])
const_n_3_2 = tf.constant([[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]])

# constant print value, shape, rank, size
tfutil.print_constant_details(const_s_0)
tfutil.print_constant_details(const_v_1_1)
tfutil.print_constant_details(const_v_1_2)
tfutil.print_constant_details(const_m_2_1)
tfutil.print_constant_details(const_m_2_2)
tfutil.print_constant_details(const_n_3_1)
tfutil.print_constant_details(const_n_3_2)


# variable
var_s_0 = tf.Variable(1)
var_v_1_1 = tf.Variable([1, 2])
var_v_1_2 = tf.Variable([1, 2, 3])
var_m_2_1 = tf.Variable([[1, 2, 3], [4, 5, 6]])
var_m_2_2 = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
var_n_3_1 = tf.Variable([[[2], [4]], [[8], [10]], [[14], [16]]])
var_n_3_2 = tf.Variable([[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]])

# variable print value, shape, rank, size
tfutil.print_variable_details(var_s_0)
tfutil.print_variable_details(var_v_1_1)
tfutil.print_variable_details(var_v_1_2)
tfutil.print_variable_details(var_m_2_1)
tfutil.print_variable_details(var_m_2_2)
tfutil.print_variable_details(var_n_3_1)
tfutil.print_variable_details(var_n_3_2)
