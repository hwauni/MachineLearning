# -*- coding: utf-8 -*-
#!/usr/bin/python
import tensorflow as tf

def tf_print_constant(const):
    sess = tf.InteractiveSession()
    print(const.eval())
    sess.close()

def tf_get_const_value(const):
    sess = tf.InteractiveSession()
    ret_const = const.eval()
    sess.close()

    return ret_const

def tf_print_variable(var):
    sess = tf.InteractiveSession()
    var.initializer.run()
    print(var.eval())
    sess.close()

def tf_get_var_value(var):
    sess = tf.InteractiveSession()
    var.initializer.run()
    ret_var = var.eval()
    sess.close()

    return ret_var

print('constant section')
const1, const2 = tf.constant([1]), tf.constant([2, 3])
print('print constant')
tf_print_constant(const1)
tf_print_constant(const2)

tmp_const1 = tf_get_const_value(const1)
tmp_const2 = tf_get_const_value(const2)
print('get constant value')
print(const1)
print(tmp_const1)
print(const2)
print(tmp_const2)

print('variable section')
var1, var2 = tf.Variable([4]), tf.Variable([5, 6])
print('print variable')
tf_print_variable(var1)
tf_print_variable(var2)
print('get variable value')
tmp_var1 = tf_get_var_value(var1)
tmp_var2 = tf_get_var_value(var2)
print(var1)
print(tmp_var1)
print(var2)
print(tmp_var2)
