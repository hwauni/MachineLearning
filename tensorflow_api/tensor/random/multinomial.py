# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    mn1 = tf.multinomial(tf.log([[0.5, 0.5]]), 10)
    tfutil.print_constant(mn1)
