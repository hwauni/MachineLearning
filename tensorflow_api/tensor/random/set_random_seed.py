# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import tfutil

if __name__ == '__main__':
    ru1 = tf.random_uniform([1])
    rn1 = tf.random_normal([1])

    print("Session 1")
    with tf.Session() as sess1:
        print(sess1.run(ru1))  # generates 'A1'
        print(sess1.run(ru1))  # generates 'A2'
        print(sess1.run(rn1))  # generates 'B1'
        print(sess1.run(rn1))  # generates 'B2'

    print("Session 2")
    with tf.Session() as sess2:
        print(sess2.run(ru1))  # generates 'A1'
        print(sess2.run(ru1))  # generates 'A2'
        print(sess2.run(rn1))  # generates 'B1'
        print(sess2.run(rn1))  # generates 'B2'

    ru2 = tf.random_uniform([1], seed=1)
    rn2 = tf.random_normal([1])

    print("Session 1")
    with tf.Session() as sess1:
        print(sess1.run(ru2))  # generates 'A1'
        print(sess1.run(ru2))  # generates 'A2'
        print(sess1.run(rn2))  # generates 'B1'
        print(sess1.run(rn2))  # generates 'B2'

    print("Session 2")
    with tf.Session() as sess2:
        print(sess2.run(ru2))  # generates 'A1'
        print(sess2.run(ru2))  # generates 'A2'
        print(sess2.run(rn2))  # generates 'B1'
        print(sess2.run(rn2))  # generates 'B2'

    tf.set_random_seed(1234)
    ru3 = tf.random_uniform([1])
    rn3 = tf.random_normal([1])

    print("Session 1")
    with tf.Session() as sess1:
        print(sess1.run(ru3))  # generates 'A1'
        print(sess1.run(ru3))  # generates 'A2'
        print(sess1.run(rn3))  # generates 'B1'
        print(sess1.run(rn3))  # generates 'B2'

    print("Session 2")
    with tf.Session() as sess2:
        print(sess2.run(ru3))  # generates 'A1'
        print(sess2.run(ru3))  # generates 'A2'
        print(sess2.run(rn3))  # generates 'B1'
        print(sess2.run(rn3))  # generates 'B2'
