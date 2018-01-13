# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil


if __name__ == '__main__':
    for xs in [(0, 0), (0, 1), (1, 0)]:
        const1, const2 = tf.constant(xs[0]), tf.constant(xs[1])

        # y : A  Tensor . Must have the same type as  x 
        y = tfutil.get_operation_value(tf.less(const1, const2))
        print(str(xs[0]) + " < " + str(xs[1]) + " -> " + str(y))
