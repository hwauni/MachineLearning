# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    const = tf.linspace(10.0, 12.0, 3, name="linspace")
    tfutil.print_constant(const)
