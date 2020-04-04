'''
Helper functions for down-scaling experiment.
'''

import numpy as np
import tensorflow as tf
from skimage.transform import resize


'''
This function extracts a gaussian kernel for given sigma and amplitude A.
Kernel extent is 3*sigma.
'''
def gaussian_kernel(sigma=1.0):

    size = np.ceil(3 * sigma)
    ij_range = np.arange(-size, size + 1)
    i, j = np.meshgrid(ij_range, ij_range, indexing='ij')
    g_kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(np.square(i) + np.square(j)) / (2 * sigma ** 2))
    g_kernel /= np.sum(g_kernel)

    g_kernel = g_kernel.astype(np.float32)

    return g_kernel


'''
This function extracts a guassian pyramid of a set of images for a given number of scales. The first scale
corresponds to the original image scale.
'''
def gaussian_pyramid(images, num_scales, use_blurring=True, use_bilinear=True, use_align_corners=False, downscale=2):

    original_size = np.array(images.shape[1:3])
    subsampled_size = np.array(original_size, copy=True)

    sigma = 2 * downscale/6.0
    gauss_kernel = tf.cast(gaussian_kernel(sigma), tf.float32)[..., tf.newaxis, tf.newaxis]

    g_pyramid = [tf.cast(images, tf.float32)]
    if images.shape[3] == 3:
        gauss_kernel = tf.tile(gauss_kernel, [1, 1, 3, 1])

    for i in range(1, 1+num_scales):
        if use_blurring:
            blurred_im = tf.nn.depthwise_conv2d(g_pyramid[i-1], gauss_kernel, strides=[1, 1, 1, 1], padding='SAME')
        else:
            blurred_im = g_pyramid[i-1]

        if use_bilinear:
            subsampled_size = subsampled_size/2
            if use_align_corners:
                g_pyramid.append(tf.image.resize_images(blurred_im, subsampled_size, align_corners=True))
            else:
                g_pyramid.append(tf.image.resize_images(blurred_im, subsampled_size, align_corners=False))
        else:
            g_pyramid.append(tf.nn.max_pool2d(input=blurred_im, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

    return g_pyramid


def gaussian_pyramid_v2(images, num_scales, sigma, use_blurring=True, use_bilinear=True, use_align_corners=False):
    original_size = np.array(images.shape)
    subsampled_size = np.array(original_size[1:3], copy=True)

    g_pyramid = [images]

    for i in range(1, 1 + num_scales):

        if use_blurring:
            gauss_kernel = tf.cast(gaussian_kernel(sigma), tf.float32)[..., tf.newaxis, tf.newaxis]
            gauss_kernel = tf.tile(gauss_kernel, [1, 1, original_size[-1], 1])

            blurred_im = tf.nn.depthwise_conv2d(g_pyramid[i - 1], gauss_kernel, strides=[1, 1, 1, 1], padding='SAME')
            sigma *= 2
        else:
            blurred_im = g_pyramid[i - 1]

        if use_bilinear:
            subsampled_size = subsampled_size / 2
            if use_align_corners:
                g_pyramid.append(tf.image.resize_images(blurred_im, subsampled_size, align_corners=True))
            else:
                g_pyramid.append(tf.image.resize_images(blurred_im, subsampled_size, align_corners=False))
        else:
            g_pyramid.append(tf.nn.max_pool(value=blurred_im, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

    return g_pyramid


'''
This function performs gaussian blurring of inpue with increased sigma. First scale corresponds to the original 
input.
'''
def gaussian_blurring(x_train, sigma, scales, mode=None):
    in_shape = x_train.shape[1:]

    blurred_images = [x_train]
    for i in range(scales):
        kernel = gaussian_kernel(sigma)
        kernel_extent = len(kernel)//2
        kernel = tf.cast(kernel[..., tf.newaxis, tf.newaxis], tf.float32)
        kernel = tf.tile(kernel, [1, 1, in_shape[-1], 1])

        if mode is not None:
            paddings = tf.constant([[0, 0], [kernel_extent, kernel_extent], [kernel_extent, kernel_extent], [0, 0]])
            blurred_images.append(tf.nn.depthwise_conv2d(tf.pad(blurred_images[0], paddings, mode="REFLECT"),
                                                         kernel, strides=[1, 1, 1, 1], padding='VALID'))
        else:
            blurred_images.append(tf.nn.depthwise_conv2d(blurred_images[0], kernel, strides=[1, 1, 1, 1], padding='SAME'))
        sigma *= 2

    return blurred_images[1:]


'''
This function extracts a 'blur-subsampling' pyramid, from paper "Making convolutional networks shift-invariant again".
'''
def antialias_pyramid(images, num_scales, kernel):

    pyramid = [tf.cast(images, tf.float32)]

    if images.shape[3] == 3:
        kernel = tf.tile(kernel, [1, 1, 3, 1])

    for i in range(1, 1+num_scales):
        blurred_im = tf.nn.max_pool2d(input=pyramid[i-1], ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        blurred_im = tf.nn.depthwise_conv2d(blurred_im, kernel, strides=[1, 1, 1, 1], padding='SAME')
        pyramid.append(tf.nn.max_pool2d(input=blurred_im, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME'))

    return pyramid


'''
Type of kernels to use for blur-subsampling, from paper "Making convolutional networks shift-invariant again".
'''
def kernels(kernel_type):

    kernel=[]
    if kernel_type == 'gaussian':
        kernel = gaussian_kernel(2*2/6.0)
    elif kernel_type == 'rectangle':
        kernel = np.ones((2, 2))
    elif kernel_type == 'triangle':
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    elif kernel_type == 'binomial':
        kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
    else:
        print("Give correct type")

    kernel = kernel/np.sum(kernel)

    return tf.cast(kernel[..., tf.newaxis, tf.newaxis], tf.float32)


'''
Function for upscaling input.
'''
def upscale(x, method='bilinear', up_factor=2):

    input_shape = x.shape[1:-1]

    # x = tf.image.resize(x, tf.math.multiply(up_factor, input_shape), method)
    x = tf.compat.v2.image.resize(x, tf.math.multiply(up_factor, input_shape), method=method)

    return x


'''
Function for resizing input.
'''
def resize_images(x, new_shape):

    x_new = []
    for c in range(len(x)):
        im = resize(x[c], (new_shape, new_shape), anti_aliasing=True)
        x_new.append(im)
    x_new = np.array(x_new, dtype='float32')

    return x_new

'''
TF function for performing gaussian blurring.
'''
def gaussian_blur(x, sigma, mode='reflect'):

    in_shape = x.shape[1:]
    kernel = gaussian_kernel(sigma)
    kernel_extent = len(kernel) // 2
    kernel = tf.cast(gaussian_kernel(sigma), tf.float32)[..., tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, in_shape[-1], 1])

    if mode == 'reflect':
        paddings = tf.constant([[0, 0], [kernel_extent, kernel_extent], [kernel_extent, kernel_extent], [0, 0]])
        x = tf.nn.depthwise_conv2d(tf.pad(x, paddings, mode='REFLECT'), kernel, strides=[1, 1, 1, 1], padding='VALID')
    else:
        x = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

    return x


'''
TF function for downscaling with gaussian blurring.
'''
def downscale_gaussian(x, sigma=0.6, down_factor=2, mode='reflect'):

    in_shape = x.shape[1:]
    kernel = gaussian_kernel(sigma)[..., np.newaxis, np.newaxis]
    kernel_extent = len(kernel) // 2
    kernel = tf.tile(kernel, [1, 1, in_shape[-1], 1])

    if mode == 'reflect':
        paddings = tf.constant([[0, 0], [kernel_extent, kernel_extent], [kernel_extent, kernel_extent], [0, 0]])
        x = tf.nn.depthwise_conv2d(tf.pad(x, paddings, mode='REFLECT'), kernel, strides=[1, 1, 1, 1], padding='VALID')
    else:
        x = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    # x = tf.image.resize_images(x, tf.div(in_shape[:-1], down_factor), align_corners=False)
    # x = tf.nn.max_pool2d(input=x, ksize=[1, 1, 1, 1], strides=[1, down_factor, down_factor, 1], padding='SAME')
    x = x[:, ::down_factor, ::down_factor, :]

    return x


'''
TF function for downscaling with built-in tf function.
'''
def downscale_tf(x, down_factor=2):

    in_shape = x.shape[1:]

    x = tf.compat.v2.image.resize(x, tf.div(in_shape[:-1], down_factor), method='gaussian', antialias=True)

    return x


'''
TF function for downscaling with blur-subsampling, from paper "Making convolutional networks shift-invariant again"..
'''
def downscale_antialias(x, sigma=0.6, down_factor=2):

    in_shape = x.shape[1:]
    kernel = gaussian_kernel(sigma)[..., np.newaxis, np.newaxis]
    kernel = tf.tile(kernel, [1, 1, in_shape[-1], 1])

    x = tf.nn.max_pool2d(input=x, ksize=[1, down_factor, down_factor, 1], strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool2d(input=x, ksize=[1, 1, 1, 1], strides=[1, down_factor, down_factor, 1], padding='SAME')

    return x


'''
TF function for downscaling without anti-alias blurring.
'''
def downscale_noblur(x, down_factor=2):

    # x = tf.image.resize_images(x, tf.div(in_shape, down_factor), align_corners=False)
    x = tf.nn.max_pool2d(input=x, ksize=[1, 1, 1, 1], strides=[1, down_factor, down_factor, 1], padding='SAME')

    return x


'''
TF function for downscaling with max-pool.
'''
def downscale_pool(x, down_factor=2):

    x = tf.nn.max_pool2d(input=x, ksize=[1, down_factor, down_factor, 1], strides=[1, down_factor, down_factor, 1],
                         padding='SAME')

    return x


'''
A wrapper function for all the downsacling methods.
'''
def downscale(x, down_factor, sigma=1, method='tf', mode='reflect'):

    if method == 'tf':
        return downscale_tf(x, down_factor)
    elif method == 'gaussian':
        return downscale_gaussian(x, sigma, down_factor, mode)
    elif method == 'noblur':
        return downscale_noblur(x, down_factor)
    elif method == 'pool':
        return downscale_pool(x, down_factor)
    else:
        return downscale_antialias(x, sigma, down_factor)
