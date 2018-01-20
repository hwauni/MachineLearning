# -*- coding: utf-8 -*-
#!/usr/bin/python

import tensorflow as tf
import numpy as np
import tfutil

n = 10
# images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100 in order
images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

const1 = tf.constant(np.array(images), dtype=tf.int32)
print(const1)
print(tf.shape(const1))
tfutil.print_constant(const1)

# Ksizes Test
# output_depth = ksize_rows * ksize_cols * depth = (1 x 1 x 1 ) = 1
# ksizes: raws: [1] col: [1]
ksizes = [1, 1, 1, 1]
strides = [1, 3, 3, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# Ksizes Test
# output_depth = ksize_rows * ksize_cols * depth = (1 x 2 x 1 ) = 2
# ksizes: raws: [1, 2]  col: [1, 2]
ksizes = [1, 1, 2, 1]
strides = [1, 3, 3, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# Ksizes Test
# output_depth = ksize_rows * ksize_cols * depth = (1 x 3 x 1 ) = 3
# ksizes: raws: [1, 2, 3]   col: [1, 2, 3]
ksizes = [1, 1, 3, 1]
strides = [1, 3, 3, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# Ksizes Test
# output_depth = ksize_rows * ksize_cols * depth = (2 x 3 x 1 ) = 6 -> [ 1  2  3 11 12 13] ..
# ksizes: raws: [1  2  3 ]   col: [1  2  3 11 12 13]
ksizes = [1, 2, 3, 1]
strides = [1, 3, 3, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# Ksizes Test
# output_depth = ksize_rows * ksize_cols * depth = (3 x 3 x 1 ) = 9 -> [ 1 2 3 11 12 13 21 22 23] ..
# ksizes: raws: [1  2  3 ]   col: [1  2  3 11 12 13 21 22 23]
ksizes = [1, 3, 3, 1]
strides = [1, 3, 3, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# strides Test
# output_depth = ksize_rows * ksize_cols * depth = (3 x 3 x 1 ) = 9 -> [ 1 2 3 11 12 13 21 22 23] ..
# strides: raws: [1, 4, 7]   col: [1, 31, 61]
ksizes = [1, 3, 3, 1]
strides = [1, 3, 3, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# strides Test
# output_depth = ksize_rows * ksize_cols * depth = (3 x 3 x 1 ) = 9 -> [ 1 2 3 11 12 13 21 22 23] ..
# strides: raws: [1, 4, 7]   col: [1, 21, 41, 61]
ksizes = [1, 3, 3, 1]
strides = [1, 2, 3, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# strides Test
# output_depth = ksize_rows * ksize_cols * depth = (3 x 3 x 1 ) = 9 -> [ 1 2 3 11 12 13 21 22 23] ..
# strides: raws: [1, 3, 5, 7]   col: [1, 31, 61]
ksizes = [1, 3, 3, 1]
strides = [1, 3, 2, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# We generate four outputs as follows:
# 1. 3x3 patches with stride length 5
ksizes = [1, 3, 3, 1]
strides = [1, 5, 5, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# 2. Same as above, but the rate is increased to 2
ksizes = [1, 3, 3, 1]
strides = [1, 5, 5, 1]
rates = [1, 2, 2, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# 3. 4x4 patches with stride length 7; only one patch should be generated
ksizes = [1, 4, 4, 1]
strides = [1, 7, 7, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='VALID')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)

# 4. Same as above, but with padding set to 'SAME'
ksizes = [1, 4, 4, 1]
strides = [1, 7, 7, 1]
rates = [1, 1, 1, 1]
exi_const1 = tf.extract_image_patches(const1, ksizes, strides, rates, padding='SAME')
print(exi_const1)
print(tf.shape(exi_const1))
tfutil.print_operation_value(exi_const1)
