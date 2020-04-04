import pickle
import re
import time

import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from tensorflow.python.keras.utils import to_categorical

from lib.plot_curves import learning_curves
from lib.helper_functions import validation_split
from models.models_segmentation import select_model, IoU
from scripts.run_BUvsTD import setup, preprocessing, finalize


'''
Script for running a toy segmentation experiment. We extract a segmentation dataset
based on Fashion-MNIST classification task. We propagate the global label based on
thresholding the pixel values. The threshold is empirically set.

This leads to a dataset of 12 classes (original 10 + background and ignore class). 
The ignore class contributes neither to loss nor to IoU computation.

To increase the complexity of the task we extract 2x2 object meshes, leading to a total 
of 150000, 25000 training and testing samples respectively. Class weights are extracted 
for dealing with class-imbalance.
'''


def extract_segmentation_mask(X, y, thres, n_classes=12):

    labels = np.zeros((X.shape[0], X.shape[1], X.shape[2], n_classes), dtype=int)
    labels[..., -1] = np.logical_and(X[..., 0] > 0, X[..., 0] <= thres)
    labels[..., -2] = X[..., 0] == 0
    labels[..., :-2] = (X > thres) * to_categorical(y)[:, np.newaxis, np.newaxis, :]

    class_priors = np.sum(labels, axis=(0, 1, 2))
    class_weights = np.sum(class_priors) / (n_classes * class_priors)
    class_weights = class_weights[y]

    return labels, class_weights


def extract_segmentation_mesh(X, y, grid_ext, shuffle=True, n_classes=12):

    train_indexes = np.arange(len(y), dtype=int)
    if shuffle:
        np.random.shuffle(train_indexes)
    train_indexes = np.reshape(train_indexes, [len(train_indexes) // grid_ext ** 2, grid_ext ** 2])

    train_data = []
    train_labels = []
    for i in train_indexes:
        data_app = []
        labels_app = []
        for j in range(grid_ext):
            data_app.append(np.concatenate(X[i[j * grid_ext:(j + 1) * grid_ext]], axis=0))
            labels_app.append(np.concatenate(y[i[j * grid_ext:(j + 1) * grid_ext]], axis=0))
        data_app = np.concatenate(data_app, axis=1)
        labels_app = np.concatenate(labels_app, axis=1)
        train_data.append(data_app)
        train_labels.append(labels_app)

    # class_weights = np.sum(train_labels) / (n_classes * np.sum(train_labels, axis=(0, 1, 2)))
    class_weights = 1 / np.sum(train_labels, axis=(0, 1, 2))
    class_weights /= class_weights[-2]
    class_weights[-1] = 0

    return np.array(train_data), np.array(train_labels), class_weights


def train(args, filepath, f_output, x_train, y_train, x_test, y_test, class_weights=None, method=None):

    base_model_name = args.model_name
    if args.extension is not None:
        base_model_name = re.sub('_' + args.extension, '', base_model_name)

    # Extracting statistics for every model-set combination and history for learning curves
    n_classes = 12
    history = []
    test_acc = np.zeros(args.repetitions)
    test_loss = np.zeros_like(test_acc)
    test_IoU = np.zeros((args.repetitions, n_classes))
    training_time = []
    inference_time = np.zeros_like(test_acc)
    callbacks = []

    print(class_weights)

    for i in range(args.repetitions):

        if args.scheduler != 'NA':

            sched = globals()[args.scheduler]
            if 'stage' in args.scheduler:
                print(args.scheduler)
                cb_decayLR = tf.keras.callbacks.LearningRateScheduler(sched(args.learning_rate, args.num_epochs),
                                                                      verbose=0)
            else:
                cb_decayLR = tf.keras.callbacks.LearningRateScheduler(sched, verbose=0)
        else:
            cb_decayLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1,
                                                              mode='auto', min_delta=0.0001, cooldown=0,
                                                              min_lr=args.learning_rate/10)

        if not callbacks:
            callbacks.append(cb_decayLR)
        else:
            callbacks[0] = cb_decayLR

        input_shape = x_train.shape[1:]
        print('Loading model: ', base_model_name)
        optimizer = tf.keras.optimizers.SGD(args.learning_rate, momentum=0.9, nesterov=True)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        if method is not None:
            model = select_model(input_shape, base_model_name, optimizer, args.weight_decay, method)
        else:
            model = select_model(input_shape, base_model_name, optimizer, args.weight_decay, class_weights=class_weights)

        x_train, y_train = shuffle(x_train, y_train)

        # Extract tranining and validation split indices
        if args.val_split != 0:
            train_ind, val_ind = validation_split(x_train, y_train, args.val_split, args.dataset == 'TOY')

        # Timing training
        start_train = time.time()

        if args.augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0,
                width_shift_range=0.1, height_shift_range=0.1, brightness_range=None, shear_range=0.0, zoom_range=0,
                channel_shift_range=0., fill_mode='nearest', cval=0., horizontal_flip=True, vertical_flip=False,
                rescale=None, preprocessing_function=None, data_format='channels_last', validation_split=0,
                dtype='float32')
            datagen.fit(x_train[train_ind])

            hist = model.fit_generator(datagen.flow(x_train[train_ind], y_train[train_ind], batch_size=args.batch_size),
                                       epochs=args.num_epochs,
                                       validation_data=(x_train[val_ind], y_train[val_ind]),
                                       callbacks=callbacks, verbose=2)

        else:
            hist = model.fit(x_train[train_ind], y=y_train[train_ind], batch_size=args.batch_size,
                             epochs=args.num_epochs,
                             verbose=2, validation_data=(x_train[val_ind], y_train[val_ind]), callbacks=callbacks)

        training_time.append(time.time() - start_train)
        history.append(hist.history)

        # Evaluate
        test_loss[i], test_acc[i], _ = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)

        start_inference = time.time()
        y_pred = model.predict(x_test, batch_size=args.batch_size, verbose=0)
        inference_time[i] = time.time() - start_inference
        test_IoU[i] = IoU(y_test, y_pred)

        if i == args.repetitions - 1:
            model.save(filepath['models'] + filepath['dataset'] + args.model_name + '.h5')

    # Store history
    with open(filepath['history'] + filepath['dataset'] + 'history_' + args.model_name + '.txt', 'wb') as f_history:
        pickle.dump(history, f_history)

    # Extract and output metrics
    mean_test_loss = np.mean(test_loss)
    std_test_loss = np.std(test_loss, ddof=1)
    mean_test_acc = np.mean(test_acc)
    std_test_acc = np.std(test_acc, ddof=1)
    mean_inference_time = np.mean(inference_time)
    std_inference_time = np.std(inference_time, ddof=1)
    mean_IoU = np.round(np.mean(test_IoU, axis=0), 4)
    std_IoU = np.round(np.std(test_IoU, axis=0, ddof=1), 4)

    # Writing statistics to file
    print("****************************************", file=f_output)
    print("Model: ", args.model_name, file=f_output)
    print("Mean Test losses:", file=f_output)
    print(f"Original scale: {mean_test_loss} +- {std_test_loss}\n", file=f_output)

    print("Mean Test accuracies:", file=f_output)
    print(f"Original scale: {mean_test_acc} +- {std_test_acc}\n", file=f_output)

    print("Mean IoU:", file=f_output)
    print(f"mean: {mean_IoU}", file=f_output)
    print(f"std: {std_IoU}\n", file=f_output)

    print(f"Mean training time: {np.mean(training_time)} +- {np.std(training_time, ddof=1)}", file=f_output)
    print("Mean inference time:", file=f_output)
    print(f"Original scale: {mean_inference_time} +- {std_inference_time}", file=f_output)
    print("****************************************\n\n\n", file=f_output)

    learning_curves(history, model_n=args.model_name, filepath=filepath['graphs'] + filepath['dataset'])


def main():

    args, filepath, f_output, _ = setup()
    x_train, x_test, y_train, y_test, _, _ = preprocessing(args)

    pad_width = 0
    x_train = np.pad(x_train, ((0, 0), (pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                     mode='constant', constant_values=0)
    x_test = np.pad(x_test, ((0, 0), (pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                    mode='constant', constant_values=0)

    y_train, y_test = np.argmax(y_train, axis=-1), np.argmax(y_test, axis=-1)

    y_train, _ = extract_segmentation_mask(x_train, y_train, thres=0.2)
    y_test, _ = extract_segmentation_mask(x_test, y_test, thres=0.2)

    x_train, y_train, class_weights = extract_segmentation_mesh(x_train, y_train, grid_ext=2)
    x_test, y_test, _ = extract_segmentation_mesh(x_test, y_test, grid_ext=2)

    train(args, filepath, f_output, x_train, y_train, x_test, y_test, class_weights=class_weights)

    finalize(f_output)


if __name__ == '__main__':
    main()

