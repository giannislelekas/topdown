import sys
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import regularizers, Model
from tensorflow.python.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Activation, add, Concatenate, \
    Softmax, MaxPool2D

from lib.blurring import upscale, downscale_gaussian, downscale_noblur, downscale_pool, downscale_antialias, \
    downscale_tf
from tensorflow.python.keras import backend as K


'''
Models for running on the toy segmentation task.
'''


def conv_block(x, n, num_filters, k_size, regularizer=None):

    for i in range(n):
        if regularizer is not None:
            x = Conv2D(filters=num_filters, kernel_size=k_size, padding='same', kernel_regularizer=regularizer)(x)
        else:
            x = Conv2D(filters=num_filters, kernel_size=k_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x

'''
Function incorporating class weights to the categorical cross entropy loss computation.
'''
def weighted_categorical_cross_entropy(y_true, y_pred, class_weights):

    return tf.math.reduce_max(y_true * class_weights, axis=-1) * K.categorical_crossentropy(y_true, y_pred)

def weighted_centropy(class_weights):
    def weighted_loss(y_true, y_pred):

        # return weighted_categorical_cross_entropy(y_true, y_pred, class_weights)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * class_weights * K.log(y_pred)
        loss = -K.sum(loss, -1)
        return loss
    return weighted_loss


'''
Function for computing the mean Intersection over Union.
'''
def mIoU(y_true, y_pred):

    #ignore last class
    y_true = y_true[..., :-1]
    y_pred = y_pred[..., :-1]

    y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)
    y_pred = tf.one_hot(y_pred, depth=11)
    IoU = tf.math.reduce_sum(y_true * y_pred, axis=[0, 1, 2]) / (
                tf.math.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1), axis=[0, 1, 2]) + 1e-32)

    return tf.math.reduce_mean(IoU, axis=-1)


'''
Function for computing the Intersection over Union.
'''
def IoU(y_true, y_pred, n_classes=12):

    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = to_categorical(y_pred, n_classes)
    IoU = np.sum(y_true * y_pred, axis=(0, 1, 2)) / (np.sum(np.logical_or(y_true, y_pred), axis=(0, 1, 2)) + 1e-32)

    return IoU


'''
Fully-Convolutional network for image segmentation from paper
'''
def FCN(input_shape, optimizer, weight_decay, class_weights, stride=4):

    def encoder(x):

        enc_1 = Conv2D(input_shape=input_shape, filters=64, kernel_size=3, padding='same',
                       kernel_regularizer=regularizers.l2(weight_decay), name='conv_1')(x)
        enc_1 = BatchNormalization(name='batch_1')(enc_1)
        enc_1 = Activation('relu')(enc_1)
        enc_1 = Conv2D(filters=64, kernel_size=3, padding='same',
                       kernel_regularizer=regularizers.l2(weight_decay))(enc_1)
        enc_1 = BatchNormalization()(enc_1)
        enc_1 = Activation('relu')(enc_1)

        max_1 = MaxPool2D(strides=2, padding='same')(enc_1)

        enc_2 = Conv2D(filters=128, kernel_size=3, padding='same',
                       kernel_regularizer=regularizers.l2(weight_decay), name='conv_2')(max_1)
        enc_2 = BatchNormalization(name='batch_2')(enc_2)
        enc_2 = Activation('relu')(enc_2)
        enc_2 = Conv2D(filters=128, kernel_size=3, padding='same',
                       kernel_regularizer=regularizers.l2(weight_decay))(enc_2)
        enc_2 = BatchNormalization()(enc_2)
        enc_2 = Activation('relu')(enc_2)

        max_2 = MaxPool2D(strides=2, padding='same')(enc_2)

        enc_4 = Conv2D(filters=256, kernel_size=3, padding='same',
                       kernel_regularizer=regularizers.l2(weight_decay), name='conv_3')(max_2)
        enc_4 = BatchNormalization(name='batch_3')(enc_4)
        enc_4 = Activation('relu')(enc_4)
        enc_4 = Conv2D(filters=256, kernel_size=3, padding='same',
                       kernel_regularizer=regularizers.l2(weight_decay))(enc_4)
        enc_4 = BatchNormalization()(enc_4)
        enc_4 = Activation('relu')(enc_4)

        max_4 = MaxPool2D(strides=2, padding='same')(enc_4)

        return max_4, max_2, max_1

    def decoder(x_4, x_2, x_1):

        x_4 = Conv2D(filters=12, kernel_size=1, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay))(x_4)
        dec_4 = Lambda(upscale, arguments={'method': interpolation, 'up_factor': 8})(x_4)
        if stride == 4:
            return dec_4

        x_2 = Conv2D(filters=12, kernel_size=1, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay))(x_2)
        up_42 = Lambda(upscale, arguments={'method': interpolation})(x_4)
        add_2 = add([up_42, x_2])
        dec_2 = Lambda(upscale, arguments={'method': interpolation, 'up_factor': 4})(add_2)
        if stride == 2:
            return dec_2

        x_1 = Conv2D(filters=12, kernel_size=1, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay))(x_1)
        up_21 = Lambda(upscale, arguments={'method': interpolation})(add_2)
        add_1 = add([up_21, x_1])
        dec_1 = Lambda(upscale, arguments={'method': interpolation, 'up_factor': 2})(add_1)

        return dec_1

    input_1 = Input(shape=input_shape)
    interpolation = 'bilinear'

    max_4, max_2, max_1 = encoder(input_1)
    out = decoder(max_4, max_2, max_1)
    out = BatchNormalization()(out)
    out = Activation('softmax')(out)

    model = Model(inputs=input_1, outputs=out)
    model.compile(optimizer=optimizer, loss=weighted_centropy(class_weights), metrics=['accuracy', mIoU])

    return model


def Unet(input_shape, optimizer, weight_decay, class_weights):

    input_1 = Input(shape=input_shape)
    interpolation = 'nearest'

    enc_1 = Conv2D(input_shape=input_shape, filters=64, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='conv_1')(input_1)
    enc_1 = BatchNormalization(name='batch_1')(enc_1)
    enc_1 = Activation('relu')(enc_1)
    enc_1 = Conv2D(filters=64, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay))(enc_1)
    enc_1 = BatchNormalization()(enc_1)
    enc_1 = Activation('relu')(enc_1)

    enc_2 = MaxPool2D(strides=2, padding='same')(enc_1)

    enc_2 = Conv2D(filters=128, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='conv_2')(enc_2)
    enc_2 = BatchNormalization(name='batch_2')(enc_2)
    enc_2 = Activation('relu')(enc_2)
    enc_2 = Conv2D(filters=128, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay))(enc_2)
    enc_2 = BatchNormalization()(enc_2)
    enc_2 = Activation('relu')(enc_2)

    enc_4 = MaxPool2D(strides=2, padding='same')(enc_2)

    enc_4 = Conv2D(filters=256, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='conv_3')(enc_4)
    enc_4 = BatchNormalization(name='batch_3')(enc_4)
    enc_4 = Activation('relu')(enc_4)
    enc_4 = Conv2D(filters=256, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay))(enc_4)
    enc_4 = BatchNormalization()(enc_4)
    enc_4 = Activation('relu')(enc_4)

    dec_2 = Lambda(upscale, arguments={'method': interpolation})(enc_4)
    dec_2 = Conv2D(filters=128, kernel_size=1, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay))(dec_2)
    dec_2 = BatchNormalization()(dec_2)
    dec_2 = Activation('relu')(dec_2)

    dec_2 = Concatenate(axis=-1)([dec_2, enc_2])
    dec_2 = Conv2D(filters=128, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='conv_5')(dec_2)
    dec_2 = BatchNormalization(name='batch_5')(dec_2)
    dec_2 = Activation('relu')(dec_2)
    dec_2 = Conv2D(filters=128, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay))(dec_2)
    dec_2 = BatchNormalization()(dec_2)
    dec_2 = Activation('relu')(dec_2)

    dec_1 = Lambda(upscale, arguments={'method' : interpolation})(dec_2)
    dec_1 = Conv2D(filters=64, kernel_size=1, padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay))(dec_1)
    dec_1 = BatchNormalization()(dec_1)
    dec_1 = Activation('relu')(dec_1)

    out = Concatenate(axis=-1)([dec_1, enc_1])
    out = Conv2D(filters=64, kernel_size=3, padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay), name='conv_7')(out)
    out = BatchNormalization(name='batch_7')(out)
    out = Activation('relu')(out)
    out = Conv2D(filters=64, kernel_size=3, padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=12, kernel_size=1, padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay), name='conv_8')(out)
    out = BatchNormalization(name='batch_8')(out)
    out = Activation('softmax')(out)

    model = Model(inputs=input_1, outputs=out)
    w_loss = weighted_centropy(class_weights)
    model.compile(optimizer=optimizer, loss=w_loss, metrics=['accuracy', mIoU])

    return model


def TD(input_shape, optimizer, weight_decay, class_weights, method='tf'):

    input_1 = Input(shape=input_shape)
    interpolation = 'nearest'

    sigma = 1

    if method == 'gaussian':
        input_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        input_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        input_2 = Lambda(downscale_noblur)(input_1)
        input_4 = Lambda(downscale_noblur)(input_2)
    elif method == 'antialias':
        input_2 = Lambda(downscale_antialias)(input_1)
        input_4 = Lambda(downscale_antialias)(input_2)
    elif method == 'tf':
        input_2 = Lambda(downscale_tf)(input_1)
        input_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        input_2 = Lambda(downscale_pool)(input_1)
        input_4 = Lambda(downscale_pool)(input_2)

    regularizer = regularizers.l2(weight_decay)

    conv_1 = conv_block(input_1, 2, num_filters=64, k_size=3, regularizer=regularizer)
    conv_2 = conv_block(input_2, 2, num_filters=128, k_size=3, regularizer=regularizer)

    x = conv_block(input_4, 2, num_filters=256, k_size=3, regularizer=regularizer)

    # 14x14
    x = Conv2D(filters=128, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = conv_block(x, 3, num_filters=128, k_size=3, regularizer=regularizer)

    x = Conv2D(filters=64, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = conv_block(x, 3, num_filters=64, k_size=3, regularizer=regularizer)

    x = Conv2D(filters=12, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)

    out = Softmax()(x)

    model = Model(inputs=input_1, outputs=out)
    w_loss = weighted_centropy(class_weights)
    model.compile(optimizer=optimizer, loss=w_loss, metrics=['accuracy', mIoU])

    return model


def select_model(input_shape, model_n, optimizer, weight_decay, class_weights, method=None):

    if model_n == 'TD':
        if method is not None:
            return TD(input_shape, optimizer, weight_decay, method, class_weights)
        else:
            return TD(input_shape, optimizer, weight_decay, class_weights)
    elif model_n == 'Unet':
        return Unet(input_shape, optimizer, weight_decay, class_weights)
    elif model_n == 'FCN':
        return FCN(input_shape, optimizer, weight_decay, class_weights, 1)

    else:
        print("Give correct model name")
        sys.exit(1)

