# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


const1 = tf.constant(1)
tfutil.print_constant(const1)
print(const1)
double1 = tf.to_double(const1)
tfutil.print_operation_value(double1)
print(double1)

const2 = tf.constant([2, 3])
tfutil.print_constant(const2)
print(const2)
double2 = tf.to_double(const2)
tfutil.print_operation_value(double2)
print(double2)

var1 = tf.Variable(4)
tfutil.print_variable(var1)
print(var1)
double3 = tf.to_double(var1)
tfutil.print_operation_value(double3)
print(double3)

var2 = tf.Variable([5, 6])
tfutil.print_variable(var2)
print(var2)
double4 = tf.to_double(var2)
tfutil.print_operation_value(double4)
print(double4)
 
