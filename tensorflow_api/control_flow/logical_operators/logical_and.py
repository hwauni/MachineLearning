# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


if __name__ == '__main__':
    # AND Gate
    print("and gate section")
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        const1, const2 = tf.constant(xs[0]), tf.constant(xs[1])

        y = tfutil.get_operation_value(tf.logical_and(tf.cast(const1, tf.bool), tf.cast(const2, tf.bool)))
        print(str(xs) + " -> " + str(tfutil.get_operation_value(tf.cast(y, tf.int32))))
        #print(str(xs) + " -> " + str(y))
