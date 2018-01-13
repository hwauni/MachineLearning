# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    const = tf.constant([[1, 2], [3, 4], [5, 6]])
    tfutil.print_constant(const)
    shuff = tf.random_shuffle(const)
    tfutil.print_constant(shuff)
