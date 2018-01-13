# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    norm = tf.random_uniform([2, 3], name="var")
    tfutil.print_constant(norm)
