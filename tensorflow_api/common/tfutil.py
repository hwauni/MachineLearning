# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf

def print_constant(const):
    sess = tf.InteractiveSession()
    print(const.eval())
    sess.close()

def print_constant_details(const):
    sess = tf.InteractiveSession()
    print(const.eval())
    print(tf.shape(const))
    print(tf.size(const))
    print(tf.rank(const))
    sess.close()

def get_const_value(const):
    sess = tf.InteractiveSession()
    ret_const = const.eval()
    sess.close()
    return ret_const

def print_variable(var):
    sess = tf.InteractiveSession()
    var.initializer.run()
    print(var.eval())
    sess.close()

def print_variable_details(var):
    sess = tf.InteractiveSession()
    var.initializer.run()
    print(var.eval())
    print(tf.shape(var))
    print(tf.size(var))
    print(tf.rank(var))
    sess.close()

def get_var_value(var):
    sess = tf.InteractiveSession()
    var.initializer.run()
    ret_var = var.eval()
    sess.close()
    return ret_var
