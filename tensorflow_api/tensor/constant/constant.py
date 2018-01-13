# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    const1 = tf.constant([1, 2, 3, 4, 5, 6, 7])
    tfutil.print_constant(const1)
    const2 = tf.constant(-1.0, shape=[2, 3])
    tfutil.print_constant(const2)

