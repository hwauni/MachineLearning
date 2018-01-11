# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

const1, const2 = tf.constant([1]), tf.constant([2, 3])
var1, var2 = tf.Variable([4]), tf.Variable([5, 6])

tfutil.print_constant(const1)
tfutil.print_constant(const2)
tfutil.print_variable(var1)
tfutil.print_variable(var2)

print('add operation section')
tfutil.print_operation_value(tf.add(const1, var1))
tfutil.print_operation_value(tf.add(const2, var2))

print('subtract operation section')
tfutil.print_operation_value(tf.subtract(const1, var1))
tfutil.print_operation_value(tf.subtract(const2, var2))

print('multiply operation section')
tfutil.print_operation_value(tf.multiply(const1, var1))
tfutil.print_operation_value(tf.multiply(const2, var2))

print('div operation section')
tfutil.print_operation_value(tf.div(const1, var1))
tfutil.print_operation_value(tf.div(const2, var2))

print('mod operation section')
tfutil.print_operation_value(tf.mod(const1, var1))
tfutil.print_operation_value(tf.mod(const2, var2))

print('square operation section')
tfutil.print_operation_value(tf.square(var1))
tfutil.print_operation_value(tf.square(var2))
tfutil.print_operation_value(tf.square(const1))
tfutil.print_operation_value(tf.square(const2))
