# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    norm1 = tf.random_normal([2, 3], mean=-1, stddev=4)
    tfutil.print_constant(norm1)

    norm2 = tf.random_normal([2, 3], seed=1234)
    tfutil.print_constant(norm2)
