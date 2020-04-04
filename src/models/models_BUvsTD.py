import sys
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import regularizers, Model
from lib.blurring import upscale, downscale_noblur, downscale_gaussian, downscale_pool, downscale_antialias, \
    downscale_tf


'''
Models used for the 1st experiment BUvsTD.
Regarding a specific architecture the BU (Bottom-Up) baseline is "Name", 
whereas the respective TD (Top-Down) is "Name_TD". The Top-Down corresponds 
to the 3rd merging method in the paper (addition + concatenation). Incorporating 
the other merging methods is trivial.

For demonstration the TD of LeNetFC using simple elementwise-addition is given
as "LeNetFC_TD_1", where "_1" corresponds to the 1st merging method of the paper.

The "*_multi" are identical to the respective "*_TD", having this time separate inputs
for each scale; they are used for assessing the input vulnerability in the 2nd, adversarial
robustness experiment.
'''


def LeNetFC(input_shape, optimizer, weight_decay):

    model = tf.keras.Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=6, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay), name='conv_1'))
    model.add(BatchNormalization(name='batch_1'))
    model.add(Activation('relu', name='act_1'))
    model.add(MaxPool2D(strides=2, padding='same'))

    model.add(Conv2D(filters=16, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay), name='conv_2'))
    model.add(BatchNormalization(name='batch_2'))
    model.add(Activation('relu', name='act_2'))
    model.add(MaxPool2D(strides=2, padding='same'))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay), name='conv_3'))
    model.add(BatchNormalization(name='batch_3'))
    model.add(Activation('relu', name='act_3'))
    model.add(Conv2D(filters=10, kernel_size=1, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay), name='conv_4'))
    model.add(BatchNormalization(name='batch_4'))
    model.add(Activation('relu', name='act_4'))

    model.add(GlobalAveragePooling2D())
    model.add(Softmax())

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def LeNetFC_TD(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    interpolation = 'nearest'

    sigma = 1

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'antialias':
        conv_2 = Lambda(downscale_antialias)(input_1)
        conv_4 = Lambda(downscale_antialias)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    regularizer = regularizers.l2(weight_decay)

    conv_1 = Conv2D(filters=6, kernel_size=5, padding='same')(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(filters=16, kernel_size=5, padding='same')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    # 7x7
    x = Conv2D(filters=32, kernel_size=3, padding='same')(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu', name='act_1')(x)

    # 14x14
    x = Conv2D(filters=16, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=16, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=5, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='act_2')(x)

    x = Conv2D(filters=6, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=6, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=6, kernel_size=5, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='act_3')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='act_4')(x)

    x = GlobalAveragePooling2D()(x)
    out = Softmax()(x)

    model = Model(inputs=input_1, outputs=out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def LeNetFC_TD_multi(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape, name='input_1')
    input_2 = Input(shape=input_shape, name='input_2')
    input_4 = Input(shape=input_shape, name='input_4')

    interpolation = 'nearest'

    sigma = 1

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_2)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_4)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_2)
        conv_4 = Lambda(downscale_noblur, arguments={'down_factor': 4})(input_4)
    elif method == 'antialias':
        conv_2 = Lambda(downscale_antialias)(input_2)
        conv_4 = Lambda(downscale_antialias)(input_4)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_2)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_4)
    else:
        conv_2 = Lambda(downscale_pool)(input_2)
        conv_4 = Lambda(downscale_pool, arguments={'down_factor': 4})(input_4)

    regularizer = regularizers.l2(weight_decay)

    conv_1 = Conv2D(filters=6, kernel_size=5, padding='same')(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(filters=16, kernel_size=5, padding='same')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    # 7x7
    x = Conv2D(filters=32, kernel_size=3, padding='same')(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 14x14
    x = Conv2D(filters=16, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=16, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=5, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=6, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=6, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=6, kernel_size=5, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    out = Softmax()(x)

    model = Model(inputs=[input_1, input_2, input_4], outputs=out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def LeNetFC_TD_uni(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    interpolation = 'nearest'

    sigma = 1

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'antialias':
        conv_2 = Lambda(downscale_antialias)(input_1)
        conv_4 = Lambda(downscale_antialias)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    regularizer = regularizers.l2(weight_decay)

    conv_1 = Conv2D(filters=16, kernel_size=5, padding='same')(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(filters=16, kernel_size=5, padding='same')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    # 7x7
    x = Conv2D(filters=16, kernel_size=3, padding='same')(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 14x14
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=16, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=5, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=16, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=5, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    out = Softmax()(x)

    model = Model(inputs=input_1, outputs=out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def LeNetFC_TD_rev(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    interpolation = 'nearest'

    sigma = 1

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'antialias':
        conv_2 = Lambda(downscale_antialias)(input_1)
        conv_4 = Lambda(downscale_antialias)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    regularizer = regularizers.l2(weight_decay)

    conv_1 = Conv2D(filters=32, kernel_size=5, padding='same')(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(filters=16, kernel_size=5, padding='same')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    # 7x7
    x = Conv2D(filters=6, kernel_size=3, padding='same')(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 14x14
    x = Conv2D(filters=16, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=16, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=5, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=32, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=32, kernel_size=3, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=5, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    out = Softmax()(x)

    model = Model(inputs=input_1, outputs=out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_light(input_shape, optimizer, weight_decay):

    model = tf.keras.Sequential()

    model.add(Conv2D(input_shape=input_shape, filters=48, kernel_size=5, padding='SAME',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=40, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=24, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='act_1'))

    model.add(MaxPool2D(pool_size=3, strides=2, padding='SAME'))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='act_2'))

    model.add(MaxPool2D(pool_size=3, strides=2, padding='SAME'))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='act_3'))
    model.add(Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='act_4'))

    model.add(GlobalAveragePooling2D())
    model.add(Softmax())

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_light_TD(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    sigma = 1
    interpolation = 'nearest'

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_2 = Conv2D(filters=48, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_1 = Conv2D(filters=48, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    # 8x8
    conv_4 = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu', name='act_1')(x)

    # 16x16
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='act_2')(x)

    # 32x32
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=40, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=24, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='act_3')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='act_4')(x)

    x = GlobalAveragePooling2D()(x)
    output = Softmax()(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_light_TD_multi(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape, name='input_1')
    input_2 = Input(shape=input_shape, name='input_2')
    input_4 = Input(shape=input_shape, name='input_4')
    sigma = 1
    interpolation = 'nearest'

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_2)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_4)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_2)
        conv_4 = Lambda(downscale_noblur, arguments={'down_factor': 4})(input_4)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_2)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_4)
    else:
        conv_2 = Lambda(downscale_pool)(input_2)
        conv_4 = Lambda(downscale_pool, arguments={'down_factor': 4})(input_4)

    conv_2 = Conv2D(filters=48, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_1 = Conv2D(filters=48, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    # 8x8
    conv_4 = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16x16
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 32x32
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=40, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=24, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    output = Softmax()(x)

    model = Model([input_1, input_2, input_4], output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_light_TD_uni(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    sigma = 1
    interpolation = 'nearest'

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_2 = Conv2D(filters=48, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_1 = Conv2D(filters=48, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    # 8x8
    conv_4 = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16x16
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 32x32
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    output = Softmax()(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_light_TD_rev(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    sigma = 1
    interpolation = 'nearest'

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_2 = Conv2D(filters=48, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_1 = Conv2D(filters=48, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    # 8x8
    conv_4 = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    x = Conv2D(filters=40, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=24, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16x16
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 32x32
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=48, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=48, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=48, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    output = Softmax()(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN(input_shape, optimizer, weight_decay):

    model = tf.keras.Sequential()

    model.add(Conv2D(input_shape=input_shape, filters=192, kernel_size=5, padding='SAME',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=160, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=96, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=3, strides=2, padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=3, strides=2, padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=10, kernel_size=1, padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())
    model.add(Softmax())

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_TD(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    sigma = 1
    interpolation = 'nearest'

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_2 = Conv2D(filters=192, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_1 = Conv2D(filters=192, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    # 8x8
    conv_4 = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16x16
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 32x32
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=160, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=96, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    output = Softmax()(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_TD_multi(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape, name='input_1')
    input_2 = Input(shape=input_shape, name='input_2')
    input_4 = Input(shape=input_shape, name='input_4')

    sigma = 1
    interpolation = 'nearest'

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_2)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_4)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_2)
        conv_4 = Lambda(downscale_noblur, arguments={'down_factor': 4})(input_4)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_2)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_4)
    else:
        conv_2 = Lambda(downscale_pool)(input_2)
        conv_4 = Lambda(downscale_pool, arguments={'down_factor': 4})(input_4)

    conv_2 = Conv2D(filters=192, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_1 = Conv2D(filters=192, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    # 8x8
    conv_4 = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    conv_4 = BatchNormalization(name='b3')(conv_4)
    conv_4 = Activation('relu')(conv_4)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16x16
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 32x32
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=160, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=96, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    output = Softmax()(x)

    model = Model([input_1, input_2, input_4], output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_TD_uni(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    sigma = 1
    interpolation = 'nearest'

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_2 = Conv2D(filters=192, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_1 = Conv2D(filters=192, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    # 8x8
    conv_4 = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16x16
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 32x32
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    output = Softmax()(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def NIN_TD_rev(input_shape, optimizer, weight_decay, method='tf'):

    input_1 = Input(shape=input_shape)
    sigma = 1
    interpolation = 'nearest'

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_2 = Conv2D(filters=192, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_1 = Conv2D(filters=192, kernel_size=5, padding='SAME',
                    kernel_regularizer=regularizers.l2(weight_decay))(input_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    # 8x8
    conv_4 = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    x = Conv2D(filters=160, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(conv_4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=96, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16x16
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_2])
    x = Concatenate(axis=-1)([x, conv_2])
    x = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 32x32
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, conv_1])
    x = Concatenate(axis=-1)([x, conv_1])
    x = Conv2D(filters=192, kernel_size=3, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=5, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=192, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=10, kernel_size=1, padding='SAME', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    output = Softmax()(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# ResNet32, for n=5
def ResNet(input_shape, optimizer, weight_decay, n=5, num_classes=10):

    def residual_block(x, out_filters, increase=False):

        strides = 1
        if increase:
            strides = 2

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        o1 = Activation('relu')(o1)
        conv1 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
        conv1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv1)

        if increase:
            proj = Conv2D(out_filters, kernel_size=1, padding='SAME', strides=strides,
                          kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
            proj = add([conv2, proj])
        else:
            proj = add([x, conv2])

        return proj

    img_input = Input(shape=input_shape)
    x = Conv2D(filters=16, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(img_input)

    for _ in range(n):
        x = residual_block(x, 16, False)

    x = residual_block(x, 32, True)
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 64, True)
    for _ in range(1, n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(img_input, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def ResNet_TD(input_shape, optimizer, weight_decay, method='tf', n=5, num_classes=10):

    sigma = 1
    interpolation = 'nearest'

    def residual_block(x, out_filters, increase=False, skip=None):

        strides = 1

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        o1 = Activation('relu')(o1)

        if increase:
            x = Conv2D(out_filters, kernel_size=1, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
            x = Lambda(upscale, arguments={'method': interpolation})(x)
            o1 = Lambda(upscale, arguments={'method': interpolation})(o1)

        conv1 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
        conv1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv1)

        proj = add([x, conv2])

        if skip is not None:
            for s in skip:
                proj = add([proj, s])
                proj = Concatenate(axis=-1)([proj, s])

            proj = BatchNormalization(momentum=0.9, epsilon=1e-5)(proj)
            proj = Activation('relu')(proj)
            proj = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                          kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(proj)

        return proj

    input_1 = Input(shape=input_shape)

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_1 = Conv2D(filters=16, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(input_1)
    conv_2 = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_2)

    x = Conv2D(filters=64, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_4)

    for _ in range(n):
        x = residual_block(x, 64, False)

    x = residual_block(x, 32, True, [conv_2])
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 16, True, [conv_1])
    for _ in range(1, n):
        x = residual_block(x, 16, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def ResNet_TD_multi(input_shape, optimizer, weight_decay, method='tf', n=5, num_classes=10):

    sigma = 1
    interpolation = 'nearest'

    def residual_block(x, out_filters, increase=False, skip=None):

        strides = 1

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        o1 = Activation('relu')(o1)

        if increase:
            x = Conv2D(out_filters, kernel_size=1, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
            x = Lambda(upscale, arguments={'method': interpolation})(x)
            o1 = Lambda(upscale, arguments={'method': interpolation})(o1)

        conv1 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
        conv1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv1)

        proj = add([x, conv2])

        if skip is not None:
            for s in skip:
                proj = add([proj, s])
                proj = Concatenate(axis=-1)([proj, s])

            proj = BatchNormalization(momentum=0.9, epsilon=1e-5)(proj)
            proj = Activation('relu')(proj)
            proj = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                          kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(proj)

        return proj

    input_1 = Input(shape=input_shape, name='input_1')
    input_2 = Input(shape=input_shape, name='input_2')
    input_4 = Input(shape=input_shape, name='input_4')

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_2)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_4)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_2)
        conv_4 = Lambda(downscale_noblur, arguments={'down_factor': 4})(input_4)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_2)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_4)
    else:
        conv_2 = Lambda(downscale_pool)(input_2)
        conv_4 = Lambda(downscale_pool, arguments={'down_factor': 4})(input_4)

    conv_1 = Conv2D(filters=16, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(input_1)
    conv_2 = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_2)

    x = Conv2D(filters=64, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_4)

    for _ in range(n):
        x = residual_block(x, 64, False)

    x = residual_block(x, 32, True, [conv_2])
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 16, True, [conv_1])
    for _ in range(1, n):
        x = residual_block(x, 16, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model([input_1, input_2, input_4], output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def ResNet_TD_uni(input_shape, optimizer, weight_decay, method='tf', n=5, num_classes=10):

    sigma = 1
    interpolation = 'nearest'

    def residual_block(x, out_filters, increase=False, skip=None):

        strides = 1

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        o1 = Activation('relu')(o1)

        if increase:
            o1 = Lambda(upscale, arguments={'method': interpolation})(o1)
            x = Lambda(upscale, arguments={'method': interpolation})(x)

        conv1 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
        conv1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv1)

        proj = add([x, conv2])

        if skip is not None:
            for s in skip:
                proj = add([proj, s])
                proj = Concatenate(axis=-1)([proj, s])

            proj = BatchNormalization(momentum=0.9, epsilon=1e-5)(proj)
            proj = Activation('relu')(proj)
            proj = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                          kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(proj)

        return proj

    input_1 = Input(shape=input_shape)

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_1 = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(input_1)
    conv_2 = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_2)

    x = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_4)

    for _ in range(n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 32, True, [conv_2])
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 32, True, [conv_1])
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def ResNet_TD_uni_multi(input_shape, optimizer, weight_decay, method='tf', n=5, num_classes=10):

    sigma = 1
    interpolation = 'nearest'

    def residual_block(x, out_filters, increase=False, skip=None):

        strides = 1

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        o1 = Activation('relu')(o1)

        if increase:
            o1 = Lambda(upscale, arguments={'method': interpolation})(o1)
            x = Lambda(upscale, arguments={'method': interpolation})(x)

        conv1 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
        conv1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv1)

        proj = add([x, conv2])

        if skip is not None:
            for s in skip:
                proj = add([proj, s])
                proj = Concatenate(axis=-1)([proj, s])

            proj = BatchNormalization(momentum=0.9, epsilon=1e-5)(proj)
            proj = Activation('relu')(proj)
            proj = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                          kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(proj)

        return proj

    input_1 = Input(shape=input_shape, name='input_1')
    input_2 = Input(shape=input_shape, name='input_2')
    input_4 = Input(shape=input_shape, name='input_4')

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_2)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_4)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_2)
        conv_4 = Lambda(downscale_noblur)(input_4)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_2)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_4)
    else:
        conv_2 = Lambda(downscale_pool)(input_2)
        conv_4 = Lambda(downscale_pool)(input_4)

    conv_1 = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(input_1)
    conv_2 = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_2)

    x = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_4)

    for _ in range(n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 32, True, [conv_2])
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 32, True, [conv_1])
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model([input_1, input_2, input_4], output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def ResNet_TD_rev(input_shape, optimizer, weight_decay, method='tf', n=5, num_classes=10):

    sigma = 1
    interpolation = 'nearest'

    def residual_block(x, out_filters, increase=False, skip=None):

        strides = 1

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        o1 = Activation('relu')(o1)

        if increase:
            x = Conv2D(out_filters, kernel_size=1, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
            x = Lambda(upscale, arguments={'method': interpolation})(x)
            o1 = Lambda(upscale, arguments={'method': interpolation})(o1)

        conv1 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
        conv1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv1)

        proj = add([x, conv2])

        if skip is not None:
            for s in skip:
                proj = add([proj, s])
                proj = Concatenate(axis=-1)([proj, s])

            proj = BatchNormalization(momentum=0.9, epsilon=1e-5)(proj)
            proj = Activation('relu')(proj)
            proj = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                          kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(proj)

        return proj

    input_1 = Input(shape=input_shape)

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_1)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_1)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_1)
        conv_4 = Lambda(downscale_noblur)(conv_2)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_1)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_1)
    else:
        conv_2 = Lambda(downscale_pool)(input_1)
        conv_4 = Lambda(downscale_pool)(conv_2)

    conv_1 = Conv2D(filters=64, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(input_1)
    conv_2 = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_2)

    x = Conv2D(filters=16, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_4)

    for _ in range(n):
        x = residual_block(x, 16, False)

    x = residual_block(x, 32, True, [conv_2])
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 64, True, [conv_1])
    for _ in range(1, n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(input_1, output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def ResNet_TD_rev_multi(input_shape, optimizer, weight_decay, method='tf', n=5, num_classes=10):

    sigma = 1
    interpolation = 'nearest'

    def residual_block(x, out_filters, increase=False, skip=None):

        strides = 1

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        o1 = Activation('relu')(o1)

        if increase:
            x = Conv2D(out_filters, kernel_size=1, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
            x = Lambda(upscale, arguments={'method': interpolation})(x)
            o1 = Lambda(upscale, arguments={'method': interpolation})(o1)

        conv1 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=strides,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(o1)
        conv1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                       kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv1)

        proj = add([x, conv2])

        if skip is not None:
            for s in skip:
                proj = add([proj, s])
                proj = Concatenate(axis=-1)([proj, s])

            proj = BatchNormalization(momentum=0.9, epsilon=1e-5)(proj)
            proj = Activation('relu')(proj)
            proj = Conv2D(out_filters, kernel_size=3, padding='SAME', strides=1,
                          kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(proj)

        return proj

    input_1 = Input(shape=input_shape, name='input_1')
    input_2 = Input(shape=input_shape, name='input_2')
    input_4 = Input(shape=input_shape, name='input_4')

    if method == 'gaussian':
        conv_2 = Lambda(downscale_gaussian, arguments={'sigma': sigma})(input_2)
        conv_4 = Lambda(downscale_gaussian, arguments={'sigma': 2*sigma, 'down_factor': 4})(input_4)
    elif method == 'noblur':
        conv_2 = Lambda(downscale_noblur)(input_2)
        conv_4 = Lambda(downscale_noblur)(input_4)
    elif method == 'tf':
        conv_2 = Lambda(downscale_tf)(input_2)
        conv_4 = Lambda(downscale_tf, arguments={'down_factor': 4})(input_4)
    else:
        conv_2 = Lambda(downscale_pool)(input_2)
        conv_4 = Lambda(downscale_pool)(input_4)

    conv_1 = Conv2D(filters=64, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(input_1)
    conv_2 = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                    kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_2)

    x = Conv2D(filters=16, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(conv_4)

    for _ in range(n):
        x = residual_block(x, 16, False)

    x = residual_block(x, 32, True, [conv_2])
    for _ in range(1, n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 64, True, [conv_1])
    for _ in range(1, n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model([input_1, input_2, input_4], output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def select_model(input_shape, model_n, optimizer, weight_decay, method=None):

    if model_n == 'LeNetFC':
        return LeNetFC(input_shape, optimizer, weight_decay)
    elif model_n == 'LeNetFC_TD':
        if method is not None:
            return LeNetFC_TD(input_shape, optimizer, weight_decay, method)
        else:
            return LeNetFC_TD(input_shape, optimizer, weight_decay)
    elif model_n == 'LeNetFC_TD_multi':
        if method is not None:
            return LeNetFC_TD_multi(input_shape, optimizer, weight_decay, method)
        else:
            return LeNetFC_TD_multi(input_shape, optimizer, weight_decay)
    elif model_n == 'LeNetFC_TD_uni':
        if method is not None:
            return LeNetFC_TD_uni(input_shape, optimizer, weight_decay, method)
        else:
            return LeNetFC_TD_uni(input_shape, optimizer, weight_decay)
    elif model_n == 'LeNetFC_TD_rev':
        if method is not None:
            return LeNetFC_TD_rev(input_shape, optimizer, weight_decay, method)
        else:
            return LeNetFC_TD_rev(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_light':
        return NIN_light(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_light_TD':
        if method is not None:
            return NIN_light_TD(input_shape, optimizer, weight_decay, method)
        else:
            return NIN_light_TD(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_light_TD_multi':
        if method is not None:
            return NIN_light_TD_multi(input_shape, optimizer, weight_decay, method)
        else:
            return NIN_light_TD_multi(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_light_TD_uni':
        if method is not None:
            return NIN_light_TD_uni(input_shape, optimizer, weight_decay, method)
        else:
            return NIN_light_TD_uni(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_light_TD_rev':
        if method is not None:
            return NIN_light_TD_rev(input_shape, optimizer, weight_decay, method)
        else:
            return NIN_light_TD_rev(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN':
        return NIN(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_TD':
        if method is not None:
            return NIN_TD(input_shape, optimizer, weight_decay, method)
        else:
            return NIN_TD(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_TD_multi':
        if method is not None:
            return NIN_TD_multi(input_shape, optimizer, weight_decay, method)
        else:
            return NIN_TD_multi(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_TD_uni':
        if method is not None:
            return NIN_TD_uni(input_shape, optimizer, weight_decay, method)
        else:
            return NIN_TD_uni(input_shape, optimizer, weight_decay)
    elif model_n == 'NIN_TD_rev':
        if method is not None:
            return NIN_TD_rev(input_shape, optimizer, weight_decay, method)
        else:
            return NIN_TD_rev(input_shape, optimizer, weight_decay)
    elif model_n == 'ResNet':
        return ResNet(input_shape, optimizer, weight_decay)
    elif model_n == 'ResNet_TD':
        if method is not None:
            return ResNet_TD(input_shape, optimizer, weight_decay, method)
        else:
            return ResNet_TD(input_shape, optimizer, weight_decay)
    elif model_n == 'ResNet_TD_multi':
        if method is not None:
            return ResNet_TD_multi(input_shape, optimizer, weight_decay, method)
        else:
            return ResNet_TD_multi(input_shape, optimizer, weight_decay)
    elif model_n == 'ResNet_TD_uni':
        if method is not None:
            return ResNet_TD_uni(input_shape, optimizer, weight_decay, method)
        else:
            return ResNet_TD_uni(input_shape, optimizer, weight_decay)
    elif model_n == 'ResNetCat_uni_multi':
        if method is not None:
            return ResNet_TD_uni_multi(input_shape, optimizer, weight_decay, method)
        else:
            return ResNet_TD_uni_multi(input_shape, optimizer, weight_decay)
    elif model_n == 'ResNet_TD_rev':
        if method is not None:
            return ResNet_TD_rev(input_shape, optimizer, weight_decay, method)
        else:
            return ResNet_TD_rev(input_shape, optimizer, weight_decay)
    elif model_n == 'ResNet_TD_rev_multi':
        if method is not None:
            return ResNet_TD_rev_multi(input_shape, optimizer, weight_decay, method)
        else:
            return ResNet_TD_rev_multi(input_shape, optimizer, weight_decay)
    else:
        print("Give correct model name")
        sys.exit(1)


