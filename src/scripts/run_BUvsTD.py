import argparse
import pickle
import os
import re
import time

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.python.keras.utils import to_categorical
from sklearn.utils import shuffle
from models.models_BUvsTD import *
from lib.plot_curves import *
from lib.helper_functions import normalization, validation_split
from lib.callbacks import scheduler_3_stage


'''
Script for running the 1st, BUvsTD experiment.
'''

'''
Commandline inputs: 

 -d MNIST
    -m LeNetFC -l 0.1 -w 0 -e 50 -r 4 -b 128 -v 0.1 -s scheduler_3_stage
    -m LeNetFC_TD -l 0.1 -w 0 -e 30 -r 4 -b 128 -v 0.1 -s scheduler_3_stage
    -m NIN_light -l 0.1 -w 0 -e 50 -r 4 -b 128 -v 0.1 -s scheduler_3_stage
    -m NIN_light_TD -l 0.1 -w 0 -e 50 -r 4 -b 128 -v 0.1 -s scheduler_3_stage
 
 -d FMNIST (Fashion-MNIST)
    -m LeNetFC -l 0.1 -w 0 -e 40 -r 4 -b 128 -v 0.1 -s scheduler_3_stage
    -m LeNetFC_TD -l 0.1 -w 0 -e 40 -r 4 -b 128 -v 0.1 -s scheduler_3_stage
    -m NIN_light -l 0.1 -w 0 -e 50 -r 4 -b 128 -v 0.1 -s scheduler_3_stage
    -m NIN_light_TD -l 0.1 -w 0 -e 50 -r 4 -b 128 -v 0.1 -s scheduler_3_stage
 
 -d CIFAR10
    -m ResNet -l 0.1 -w 5e-4 -e 100 -r 4 -b 128 -v 0.1 -s scheduler_3_stage -p True
    -m ResNet_TD -l 0.1 -w 5e-4 -e 100 -r 4 -b 128 -v 0.1 -s scheduler_3_stage -p True
    -m NIN -l 0.1 -w 5e-4 -e 100 -r 4 -b 128 -v 0.1 -s scheduler_3_stage -p True
    -m NIN_TD -l 0.1 -w 5e-4 -e 100 -r 4 -b 128 -v 0.1 -s scheduler_3_stage -p True
    
    augmented cases:
    -m ResNet -l 0.1 -w 1e-4 -e 200 -r 4 -b 128 -v 0.1 -s scheduler_3_stage -p True -a True -ex aug 
    -m ResNet_TD -l 0.1 -w 1e-4 -e 200 -r 4 -b 128 -v 0.1 -s scheduler_3_stage -p True -a True -ex aug
    -m NIN -l 0.1 -w 1e-4 -e 200 -r 4 -b 128 -v 0.1 -s scheduler_3_stage -p True -a True -ex aug
    -m NIN_TD -l 0.1 -w 1e-4 -e 200 -r 4 -b 128 -v 0.1 -s scheduler_3_stage -p True -a True -ex aug
    
NOTE: The _uni and _rev TD variants share the inputs of the corresponding TD.
'''


def parse_input():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, metavar='STRING',
                        choices=('TOY', 'MNIST', 'MNIST_S', 'FMNIST', 'FMNIST_S', 'CIFAR10', 'IMAGENETTE'),
                        help='dataset, choices {%(choices)s}')
    parser.add_argument('-m', '--model_name', type=str, metavar='STRING', help='model name')
    parser.add_argument('-l', '--learning_rate', type=float, metavar='NUMBER', help='base learning rate')
    parser.add_argument('-w', '--weight_decay', type=float, metavar='NUMBER', help='regularization weight decay')
    parser.add_argument('-e', '--num_epochs', type=int, metavar='NUMBER', help='number of training epochs')
    parser.add_argument('-r', '--repetitions', type=int, metavar='NUMBER', default=1, help='number of repetitions')
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default:128)')
    parser.add_argument('-v', '--val_split', type=float, default=128, metavar='NUMBER', help='validation split ratio')
    parser.add_argument('-g', '--gpus', type=int, default=1, metavar='NUMBER', help='number of GPUs')
    parser.add_argument('-s', '--scheduler', type=str, help='call to a scheduler for LR')
    parser.add_argument('-n', '--normalization_type', type=str, metavar='STRING', default='no',
                        choices=('no', 'max', 'min_max', 'zero_unit', 't_pos'),
                        help='normalization type, choices {%(choices)s}')
    parser.add_argument('-p', '--pixel_mean', type=bool, default=False, metavar='BOOL', help='subtract pixel-mean')
    parser.add_argument('-us', '--use_std', type=bool, default=False, metavar='BOOL', help='use std')
    parser.add_argument('-mv', '--multival', type=bool, default=False, metavar='BOOL', help='validate on multiple sets')
    parser.add_argument('-u', '--upscale', type=bool, default=False, metavar='BOOL', help='restore to original scale')
    parser.add_argument('-a', '--augmentation', type=bool, default=False, metavar='BOOL', help='use augmentation')
    parser.add_argument('-ex', '--extension', type=str, metavar='STRING', default=None, help='name extension')
    parser.add_argument('-ab', '--adversarial_batch_size', type=int, default=None, metavar='NUMBER',
                        help='adversarial batch size (if none one batch)')
    parser.add_argument('-abl', '--ablation', type=bool, default=False, metavar='BOOL', help='ablation')
    parser.add_argument('-t', '--target_class', type=int, default=None, help='target class for targeted attack')
    parser.add_argument('-sh', '--shuffle', type=bool, default=False, metavar='BOOL', help='shuffle')
    parser.add_argument('-im', '--class_imbalance', type=bool, default=False, metavar='BOOL', help='class imbalance')

    args = parser.parse_args()

    return args


def print_args(args, f_output):

    args_dict = args.__dict__
    keys = args_dict.keys()

    print("****************************************", file=f_output)
    print("****************************************", file=f_output)
    for key in keys:
        print(f"{key}: {args_dict[key]}", file=f_output)
    print("****************************************", file=f_output)
    print("****************************************\n\n\n", file=f_output)


def setup():

    args = parse_input()
    out_path = './../../output/'
    filepath = {'models': out_path + 'models/',
                'history': out_path + 'history/',
                'output': out_path + 'output/',
                'graphs': out_path + 'graphs/'}

    if args.extension is not None:
        args.model_name = args.model_name + '_' + args.extension

    filepath['dataset'] = args.dataset + '/'
    if args.dataset == 'TOY' or args.dataset == 'CIFAR10':
        orig_size = [32, 32]
    elif args.dataset == 'IMAGENETTE':
        orig_size = [128, 128]
    else:
        orig_size = [28, 28]

    directory = filepath['models'] + filepath['dataset']
    if not os.path.exists(directory):
        print(f"Generating folder {directory}")
        os.makedirs(directory)

    directory = filepath['history'] + filepath['dataset']
    if not os.path.exists(directory):
        print(f"Generating folder {directory}")
        os.makedirs(directory)

    directory = filepath['output'] + filepath['dataset']
    if not os.path.exists(directory):
        print(f"Generating folder {directory}")
        os.makedirs(directory)

    directory = filepath['graphs'] + filepath['dataset']
    if not os.path.exists(directory):
        print(f"Generating folder {directory}")
        os.makedirs(directory)

    f_output = open(filepath['output'] + filepath['dataset'] + 'output_' + args.model_name + '.txt', 'w+')

    print_args(args, f_output)

    return args, filepath, f_output, orig_size


def preprocessing(args):

    if args.dataset == 'MNIST' or args.dataset == 'MNIST_S':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif args.dataset == 'FMNIST' or args.dataset == 'FMNIST_S':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    n_classes = len(unique_labels(y_train, y_test))
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)

    # 4D datasets for tf
    if len(x_train.shape) == 3:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    # Normalize
    if x_train.dtype != 'float32':
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.0
        x_test /= 255.0

    if args.pixel_mean:
        x_train_mean = np.mean(x_train, 0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    x_train = normalization(args.normalization_type, train=x_train)
    x_test = normalization(args.normalization_type, train=x_test)

    # Extract tf dataset
    x_test_dset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)
    test_steps = int(np.ceil(len(x_test) / args.batch_size))

    print("Training dataset shape: ", x_train.shape)
    print("Testing dataset shape: ", x_test.shape, "\n")

    return x_train, x_test, y_train, y_test, x_test_dset, test_steps


def train(args, filepath, f_output, model_n, y_train, y_test, test_steps, x_test_dset=None, x_train=None,
          method=None, save_all_weights=False):

    print(model_n)
    base_model_name = args.model_name
    if args.extension is not None:
        base_model_name = re.sub('_' + args.extension, '', base_model_name)

    # Extracting statistics for every model-set combination and history for learning curves
    history = []
    test_acc = np.zeros(args.repetitions)
    test_loss = np.zeros_like(test_acc)
    training_time = []
    inference_time = np.zeros_like(test_acc)
    callbacks = []

    n_classes = len(y_train[0])
    y_test = np.argmax(y_test, axis=1)
    agg_cm = np.zeros((n_classes, n_classes))

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
            cb_decayLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1,
                                                              mode='auto', min_delta=0.0001, cooldown=0,
                                                              min_lr=args.learning_rate / 100)

        if not callbacks:
            callbacks.append(cb_decayLR)
        else:
            callbacks[0] = cb_decayLR

        # Resetting the model for the next iteration
        input_shape = x_train.shape[1:]
        print('Loading model: ', base_model_name)
        optimizer = tf.keras.optimizers.SGD(args.learning_rate, momentum=0.9, nesterov=True)

        if method is not None:
            print("Method: ", method)
            model = select_model(input_shape, base_model_name, optimizer, args.weight_decay, method)
        else:
            model = select_model(input_shape, base_model_name, optimizer, args.weight_decay)

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
            if args.val_split != 0:
                hist = model.fit(x_train[train_ind], y=y_train[train_ind], batch_size=args.batch_size, epochs=args.num_epochs,
                                 verbose=2, validation_data=(x_train[val_ind], y_train[val_ind]), callbacks=callbacks)
            else:
                hist = model.fit(x_train, y=y_train, batch_size=args.batch_size,
                                 epochs=args.num_epochs,
                                 verbose=2, validation_data=x_test_dset, validation_steps=test_steps,
                                 callbacks=callbacks)

        training_time.append(time.time() - start_train)
        history.append(hist.history)

        test_loss[i], test_acc[i] = model.evaluate(x_test_dset, steps=test_steps, verbose=0)

        start_inference = time.time()
        y_pred = model.predict(x_test_dset, steps=test_steps, verbose=0)
        inference_time[i] = time.time() - start_inference

        # From one-hot to single class prediction
        y_pred = np.argmax(y_pred, axis=1)
        agg_cm += confusion_matrix(y_test, y_pred)

        if save_all_weights:
            model.save(filepath['models'] + filepath['dataset'] + model_n + '_it' + str(i) + '.h5')
        # Checkpoint model in last iter
        else:
            if i == args.repetitions - 1:
                model.save(filepath['models'] + filepath['dataset'] + model_n + '.h5')

    # Store history
    with open(filepath['history'] + filepath['dataset'] + 'history_' + model_n + '.txt', 'wb') as f_history:
        pickle.dump(history, f_history)

    # Extract and output metrics
    mean_test_loss = np.mean(test_loss)
    std_test_loss = np.std(test_loss, ddof=1)
    mean_test_acc = np.mean(test_acc)
    std_test_acc = np.std(test_acc, ddof=1)
    mean_inference_time = np.mean(inference_time)
    std_inference_time = np.std(inference_time, ddof=1)

    agg_cm /= args.repetitions
    if True:
        agg_cm = np.round(agg_cm/np.sum(agg_cm, axis=1), 3)

    # Writing statistics to file
    print("****************************************", file=f_output)
    print("Model: ", model_n, file=f_output)
    print(f"Mean test loss: {mean_test_loss} +- {std_test_loss}", file=f_output)
    print(f"Mean test accuracy: {mean_test_acc} +- {std_test_acc}\n", file=f_output)

    print("Aggregated confusion matrix:", file=f_output)
    print(f"{agg_cm}\n", file=f_output)

    print(f"Mean training time: {np.mean(training_time)} +- {np.std(training_time, ddof=1)}", file=f_output)
    print(f"Mean inference time: {mean_inference_time} +- {std_inference_time}", file=f_output)
    print("****************************************\n\n\n", file=f_output)

    learning_curves(history, model_n=model_n, filepath=filepath['graphs'] + filepath['dataset'])


def finalize(f_output):

    f_output.close()

    return


def main():

    args, filepath, f_output, orig_size = setup()

    print(f"Executing {args.model_name} on dataset {args.dataset} \n")

    # Timing the execution time for the completion of the experiment
    start = time.time()

    x_train, x_test, y_train, y_test, x_test_dset, test_steps = preprocessing(args)
    train(args, filepath, f_output, model_n=args.model_name, y_train=y_train, y_test=y_test, test_steps=test_steps,
          x_test_dset=x_test_dset, x_train=x_train, method='tf', save_all_weights=True)

    print(f"Total duration of the experiment: {time.time() - start}", file=f_output)
    finalize(f_output)


if __name__ == "__main__":
    main()
