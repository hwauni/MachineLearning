# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    start = 3
    limit = 18
    delta = 3
    const1 = tf.range(start, limit, delta)
    tfutil.print_constant(const1)

    limit = 5
    const2 = tf.range(limit)
    tfutil.print_constant(const2)
