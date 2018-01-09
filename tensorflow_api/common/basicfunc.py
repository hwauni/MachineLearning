# -*- coding: utf-8 -*-
#!/usr/bin/python
# basicfunc.py

import tensorflow as tf
import functools, operator

def get_length(t):
    temp = (dim.value for dim in t.get_shape())         # dim is Dimension class.
    return functools.reduce(operator.mul, temp)

def show_constant(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    sess.close()

def show_constant_detail(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    print('shape :', tf.shape(t))
    print('size  :', tf.size(t))
    print('rank  :', tf.rank(t))
    print(t.get_shape())

    sess.close()

def show_variable(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    print(v.eval())
    sess.close()

def var2_numpy(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    n = v.eval()
    sess.close()

    return n

def op2_numpy(op):
    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)
    ret = sess.run(op)
    sess.close()

    return ret

def show_operation(op):
    print(op2_numpy(op))
