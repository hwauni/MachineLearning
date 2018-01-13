# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


if __name__ == '__main__':
    for xs in [0, 1]:
        const = tf.constant(xs)

        y = tfutil.get_operation_value(tf.logical_not(tf.cast(const, tf.bool)))
        print(str(xs) + " -> " + str(tfutil.get_operation_value(tf.cast(y, tf.int32))))
        #print(str(xs) + " -> " + str(y))
