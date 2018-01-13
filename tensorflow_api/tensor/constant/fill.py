# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    const = tf.fill([2, 3], 9)
    tfutil.print_constant(const)

