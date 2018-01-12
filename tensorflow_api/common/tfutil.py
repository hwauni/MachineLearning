# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf

def print_constant(const):
    sess = tf.InteractiveSession()
    print(const.eval())
    sess.close()

def print_constant_details(const):
    sess = tf.InteractiveSession()
    print("constant details")
    print(const.eval())
    print("shape" , sess.run(tf.shape(const)), 
        "rank" , sess.run(tf.rank(const)),
        "size" , sess.run(tf.size(const)))
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
    print("variable details")
    print(var.eval())
    print("shape" , sess.run(tf.shape(var)), 
        "rank" , sess.run(tf.rank(var)),
        "size" , sess.run(tf.size(var)))
    sess.close()

def get_var_value(var):
    sess = tf.InteractiveSession()
    var.initializer.run()
    ret_var = var.eval()
    sess.close()
    return ret_var

def get_operation_value(op):
    sess = tf.InteractiveSession()
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)
    ret_opt = sess.run(op)
    sess.close()
    return ret_opt

def print_operation_value(op):
    print(get_operation_value(op))
