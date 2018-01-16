# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


const1 = tf.constant(127, dtype=tf.int32)
tfutil.print_constant(const1)
print(const1)
f32 = tf.saturate_cast(const1, tf.float32)
tfutil.print_operation_value(f32)
print(f32)
f64 = tf.saturate_cast(const1, tf.float64)
tfutil.print_operation_value(f64)
print(f64)
i8 = tf.saturate_cast(const1, tf.int8)
tfutil.print_operation_value(i8)
print(i8)
i16 = tf.saturate_cast(const1, tf.int16)
tfutil.print_operation_value(i16)
print(i16)
i64 = tf.saturate_cast(const1, tf.int64)
tfutil.print_operation_value(i64)
print(i64)
u8 = tf.saturate_cast(const1, tf.uint8)
tfutil.print_operation_value(u8)
print(u8)

''' 
# not support
s = tf.saturate_cast(const1, tf.string)
tfutil.print_operation_value(s)
print(s)
'''

const2 = tf.constant([127, 255])
tfutil.print_constant(const2)
print(const2)
f32 = tf.saturate_cast(const2, tf.float32)
tfutil.print_operation_value(f32)
print(f32)
f64 = tf.saturate_cast(const2, tf.float64)
tfutil.print_operation_value(f64)
print(f64)
i8 = tf.saturate_cast(const2, tf.int8)
tfutil.print_operation_value(i8)
print(i8)
i16 = tf.saturate_cast(const2, tf.int16)
tfutil.print_operation_value(i16)
print(i16)
i64 = tf.saturate_cast(const2, tf.int64)
tfutil.print_operation_value(i64)
print(i64)
u8 = tf.saturate_cast(const2, tf.uint8)
tfutil.print_operation_value(u8)
print(u8)


var1 = tf.Variable(255, dtype=tf.int32)
tfutil.print_variable(var1)
print(var1)
f32 = tf.saturate_cast(var1, tf.float32)
tfutil.print_operation_value(f32)
print(f32)
f64 = tf.saturate_cast(var1, tf.float64)
tfutil.print_operation_value(f64)
print(f64)
i8 = tf.saturate_cast(var1, tf.int8)
tfutil.print_operation_value(i8)
print(i8)
i16 = tf.saturate_cast(var1, tf.int16)
tfutil.print_operation_value(i16)
print(i16)
i64 = tf.saturate_cast(var1, tf.int64)
tfutil.print_operation_value(i64)
print(i64)
u8 = tf.saturate_cast(var1, tf.uint8)
tfutil.print_operation_value(u8)
print(u8)

var2 = tf.Variable([256, 65536], dtype=tf.int32)
tfutil.print_variable(var2)
print(var2)

f32 = tf.saturate_cast(var2, tf.float32)
tfutil.print_operation_value(f32)
print(f32)
f64 = tf.saturate_cast(var2, tf.float64)
tfutil.print_operation_value(f64)
print(f64)
i8 = tf.saturate_cast(var2, tf.int8)
tfutil.print_operation_value(i8)
print(i8)
i16 = tf.saturate_cast(var2, tf.int16)
tfutil.print_operation_value(i16)
print(i16)
i64 = tf.saturate_cast(var2, tf.int64)
tfutil.print_operation_value(i64)
print(i64)
u8 = tf.saturate_cast(var2, tf.uint8)
tfutil.print_operation_value(u8)
print(u8)
