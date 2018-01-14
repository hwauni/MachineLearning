# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


if __name__ == '__main__':
    const1 = tf.constant(1, dtype=tf.float32)
    tfutil.print_constant(const1)
    print(const1)
    bfloat1 = tf.to_bfloat16(const1)
    tfutil.print_operation_value(bfloat1)
    print(bfloat1)

    const2 = tf.constant([2, 3], dtype=tf.float32)
    tfutil.print_constant(const2)
    print(const2)
    bfloat2 = tf.to_bfloat16(const2)
    tfutil.print_operation_value(bfloat2)
    print(bfloat2)

    var1 = tf.Variable(4, dtype=tf.float32)
    tfutil.print_variable(var1)
    print(var1)
    bfloat3 = tf.to_bfloat16(var1)
    tfutil.print_operation_value(bfloat3)
    print(bfloat3)

    var2 = tf.Variable([5, 6], dtype=tf.float32)
    tfutil.print_variable(var2)
    print(var2)
    bfloat4 = tf.to_bfloat16(var2)
    tfutil.print_operation_value(bfloat4)
    print(bfloat4)
 
