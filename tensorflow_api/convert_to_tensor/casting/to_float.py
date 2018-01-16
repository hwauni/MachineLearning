# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


const1 = tf.constant(1)
tfutil.print_constant(const1)
print(const1)
float1 = tf.to_float(const1)
tfutil.print_operation_value(float1)
print(float1)

const2 = tf.constant([2, 3])
tfutil.print_constant(const2)
print(const2)
float2 = tf.to_float(const2)
tfutil.print_operation_value(float2)
print(float2)

var1 = tf.Variable(4)
tfutil.print_variable(var1)
print(var1)
float3 = tf.to_float(var1)
tfutil.print_operation_value(float3)
print(float3)

var2 = tf.Variable([5, 6])
tfutil.print_variable(var2)
print(var2)
float4 = tf.to_float(var2)
tfutil.print_operation_value(float4)
print(float4)
 
