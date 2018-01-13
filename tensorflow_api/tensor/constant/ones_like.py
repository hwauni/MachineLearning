# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    const = tf.constant([[1, 2, 3], [1, 2, 3]])
    tfutil.print_constant(const)
    const = tf.ones_like(const)
    tfutil.print_constant(const)

