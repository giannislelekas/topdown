import sys

from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import regularizers, Model, Input
from tensorflow.python.keras.layers import BatchNormalization, Activation, Conv2D, add, GlobalAveragePooling2D, Dense, \
    MaxPool2D, Lambda, Concatenate

from lib.blurring import upscale, downscale


'''
Models for training on Imagenette.
'''

def ResNet18(input_shape, optimizer, weight_decay, num_classes=10, gpus=1):

    def residual_block(x, out_filters, increase=False, name=None):

        strides = 1
        if increase:
            strides = 2

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        if name is not None:
            o1 = Activation('relu', name=name)(o1)
        else:
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
    x = Conv2D(filters=32, kernel_size=3, padding='SAME',
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(img_input)
    x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    x = residual_block(x, 32, False, name='act_1')
    for _ in range(1, 2):
        x = residual_block(x, 32, False)

    x = residual_block(x, 64, True, name='act_2')
    for _ in range(1, 2):
        x = residual_block(x, 64, False)

    x = residual_block(x, 128, True, name='act_3')
    for _ in range(1, 2):
        x = residual_block(x, 128, False)

    x = residual_block(x, 256, True, name='act_4')
    for _ in range(1, 2):
        x = residual_block(x, 256, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu', name='act_5')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(img_input, output)
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def ResNet18_TD(input_shape, optimizer, weight_decay, num_classes=10, method='tf', gpus=1):

    def residual_block(x, out_filters, increase=False, skip=None, name=None):

        strides = 1

        o1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        if name is not None:
            o1 = Activation('relu', name=name)(o1)
        else:
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
    interpolation = 'nearest'

    skip = Lambda(downscale, arguments={'down_factor': 8, 'method': method})(input_1)
    skip = Conv2D(filters=256, kernel_size=3, padding='SAME', strides=1,
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(skip)

    x = Lambda(downscale, arguments={'down_factor': 16, 'method': method})(input_1)
    x = Conv2D(filters=256, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(x)
    x = Lambda(upscale, arguments={'method': interpolation})(x)
    x = add([x, skip])
    x = Concatenate(axis=-1)([x, skip])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='SAME', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(x)

    # 1st block
    x = residual_block(x, 256, False, name='act_1')
    for _ in range(1, 2):
        x = residual_block(x, 256, False)

    # 2nd block
    skip = Lambda(downscale, arguments={'down_factor': 4, 'method': method})(input_1)
    skip = Conv2D(filters=128, kernel_size=3, padding='SAME', strides=1,
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(skip)
    x = residual_block(x, 128, True, [skip], name='act_2')
    for _ in range(1, 2):
        x = residual_block(x, 128, False)

    # 3rd block
    skip = Lambda(downscale, arguments={'down_factor': 2, 'method': method})(input_1)
    skip = Conv2D(filters=64, kernel_size=3, padding='SAME', strides=1,
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(skip)
    x = residual_block(x, 64, True, [skip], name='act_3')
    for _ in range(1, 2):
        x = residual_block(x, 64, False)

    # 4th bloock
    skip = Conv2D(filters=32, kernel_size=3, padding='SAME', strides=1,
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(input_1)
    x = residual_block(x, 32, True, [skip], name='act_4')
    for _ in range(1, 2):
        x = residual_block(x, 32, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu', name='act_5')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(input_1, output)
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def select_model(input_shape, model_n, optimizer, weight_decay, method=None, gpus=1):

    if model_n == 'ResNet18':
        return ResNet18(input_shape, optimizer, weight_decay, gpus=gpus)
    elif model_n == 'ResNet18_TD':
        if method is not None:
            return ResNet18_TD(input_shape, optimizer, weight_decay, method, gpus=gpus)
        else:
            return ResNet18_TD(input_shape, optimizer, weight_decay, gpus=gpus)
    else:
        print("Give correct model name")
        sys.exit(1)


