import json
import os
import glob
import pickle
import re
import time
import wget
import tarfile
import numpy as np
import tensorflow as tf

import matplotlib.image as mpimg
from skimage.transform import resize
from sklearn.decomposition import PCA
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import Sequence, to_categorical

from lib.plot_curves import learning_curves
from scripts.run_BUvsTD import setup
from models.models_imagenette import select_model
from lib.callbacks import ConfusionMatrixCB, scheduler_3_stage


'''
Script for training on Imagenette dataset.
'''

'''
Commandline inputs: 

 -d IMAGENETTE:
    -m ResNet18 -l 0.1 -w 1e-3 -e 50 -r 1 -b 128 -s scheduler_3_stage -p True
    -m ResNet18_TD -l 0.05 -w 1e-3 -e 50 -r 1 -b 64 -s scheduler_3_stage -p True
'''

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


class ImagenetteGenerator_inmem(Sequence):

    def __init__(self, X, y, batch_size, shuffle=True, crop_size=128, val=False):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_size = crop_size
        self.val = val

        if not self.val:
            self.statistics = self.extract_statistics(self.X)

        self.augmenter = ImageDataGenerator(horizontal_flip=True)
        self.indexes = np.arange(len(self.X), dtype=int)

        self.on_epoch_end()

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def extract_statistics(self, x):

        statistics = {}
        in_shape = x.shape
        x = x.reshape(-1, in_shape[1] * in_shape[2], in_shape[-1])

        statistics['mean'] = np.mean(x, axis=1, keepdims=True)
        statistics['std'] = np.std(x, axis=1, keepdims=True)

        x = (x - statistics['mean']) / statistics['std']

        cov_n = max(x.shape[1] - 1, 1)
        cov = np.matmul(np.swapaxes(x, -1, -2), x) / cov_n

        statistics['U'], statistics['S'], statistics['V'] = np.linalg.svd(cov)

        return statistics

    def pca_aug(self, x, index):

        in_shape = x.shape
        res_shape = (in_shape[0], in_shape[1]*in_shape[2], in_shape[3])
        alphas = np.random.randn(*self.statistics['S'][index].shape) * 0.1

        delta = np.squeeze(np.matmul(self.statistics['U'][index], np.expand_dims(alphas * self.statistics['S'][index], axis=-1)))
        delta = np.expand_dims(delta, axis=1)
        delta = delta * self.statistics['std'][index]

        delta = np.broadcast_to(delta, res_shape)
        delta = delta.reshape(-1, *in_shape[1:])
        x_aug = x + delta

        return x_aug

    def __len__(self):

        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, item):

        index = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]

        x = self.X[index]
        y = self.y[index]

        if not self.val:
            x = self.pca_aug(x, index)
            x = self.augmenter.flow(x, batch_size=len(x), shuffle=False).next()
            xc = []

            for img in x:
                xc.append(random_crop(img, (self.crop_size, self.crop_size)))

            x = np.array(xc, dtype=np.float32)

        return x, to_categorical(y, 10)


class ImagenetteGenerator(Sequence):

    def __init__(self, root_dir, dset_dir, image_format, batch_size, new_shape=128,
                 res_shape=156, channels=3, num_classes=10, shuffle=True, statistics=None):

        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            self.download_files()
        self.dset_dir = dset_dir
        self.image_format = image_format
        self.batch_size = batch_size
        self.res_shape = res_shape
        self.new_shape = new_shape
        self.channels = channels
        self.num_classes = num_classes
        self.shuffle = shuffle

        self.augmenter = ImageDataGenerator(horizontal_flip=True)

        self.image_filenames = []
        self.class_mapping = {}
        self.labels = []
        self.get_image_filenames()

        if statistics is None:
            X = self.retrieve_set()
            self.statistics = self.extract_statistics(X)
        else:
            self.statistics = statistics

        self.on_epoch_end()

    def download_files(self):

        if 'woof' in self.root_dir:
            dataset = 'imagewoof2'
            print('Downloading Imagewoof')
            wget.download('https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz', re.sub(dataset + '/', '', self.root_dir))
        else:
            dataset = 'imagenette2'
            print('Downloading Imagenette2')
            wget.download('https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz', re.sub(dataset + '/', '', self.root_dir))

        print('Downloading complete')

        print('Extracting files')
        tar = tarfile.open(self.root_dir[:-1] + '.tgz', "r:gz")
        tar.extractall(path=re.sub(dataset + '/', '', self.root_dir))
        tar.close()
        print('Extracting complete')

        wget.download('https://raw.githubusercontent.com/ozendelait/wordnet-to-json/master/mapping_imagenet.json',
                      self.root_dir)

    def load_json(self, filepath):

        with open(filepath, 'r') as f:
            return json.load(f)

    def load_img(self, filename):

        img = mpimg.imread(filename)
        if len(img.shape) < 3:
            img = np.tile(img[..., np.newaxis], [1, 1, self.channels])

        return img

    def retrieve_set(self):

        X, y = [], []

        for filename, label in zip(self.image_filenames, self.labels):
            img = mpimg.imread(filename)
            img = img.astype(np.float32) / 255.0
            if len(img.shape) < 3:
                img = np.tile(img[..., np.newaxis], [1, 1, self.channels])
            img = resize(img, [self.res_shape, self.res_shape, self.channels], anti_aliasing=True, mode='reflect')

            X.append(img)
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype='uint8')

        np.savez(self.dset_dir + 'data.npz', X, y, self.class_mapping)

        return X, y

    def extract_statistics(self, X):

        statistics = {}

        statistics['max'] = np.max(X)
        if statistics['max'] > 1:
            X /= statistics['max']

        statistics['mean'] = np.mean(X, axis=0)
        statistics['std'] = np.std(X, axis=0, ddof=1)

        pca = PCA(n_components=3)
        pca.fit(np.reshape(X - statistics['mean'], [len(X), np.prod(X.shape[1:])]))

        statistics['eig_vec'] = np.transpose(np.reshape(pca.components_, [3, X.shape[1], X.shape[1], 3]),
                                             axes=(1, 2, 3, 0))
        statistics['eig_val'] = pca.explained_variance_

        np.save(self.root_dir + 'statistics.npy', statistics)

        return statistics

    def get_image_filenames(self):

        files = np.array(os.listdir(self.dset_dir))
        sorted_ind = np.argsort([int(file[1:]) for file in files])
        files = files[sorted_ind]

        if not self.class_mapping:
            mapping = self.load_json(self.root_dir + 'mapping_imagenet.json')
            c = 0
            for file in files:
                for _, j in enumerate(mapping):
                    if j['v3p0'] == file:
                        self.class_mapping[c] = j['label'].split(',')[0]
                        c += 1
                    if c == len(files):
                        break

        c = 0
        for file in files:
            file = file.strip()
            image_paths = glob.glob(os.path.join(self.dset_dir, file, "*." + self.image_format))
            if image_paths:
                self.image_filenames.extend(image_paths)
                self.labels.extend(c * np.ones(len(image_paths), dtype='uint8'))
            c += 1

        self.image_filenames = np.array(self.image_filenames)
        self.labels = np.array(self.labels)

    def __len__(self):

        return int(np.ceil(len(self.labels)/self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_im_filenames = [self.image_filenames[k] for k in indexes]

        X = []
        for filename in list_im_filenames:
            img = mpimg.imread(filename)
            if len(img.shape) < 3:
                img = np.tile(img.astype(np.float32)[..., np.newaxis], [1, 1, self.channels])

            img = resize(img, [self.res_shape, self.res_shape, self.channels], anti_aliasing=True, mode='reflect')
            if np.max(img) > 1:
                img /= self.statistics['max']
            if 'val' not in self.dset_dir:
                img += np.matmul(self.statistics['eig_vec'],
                                 np.random.normal(scale=0.1, size=3)*self.statistics['eig_val'])

                if np.min(img) < 0:
                    img -= np.min(img)
                img = np.clip(img, 0, 1)

                img = random_crop(img, (self.new_shape, self.new_shape))
            X.append(img)

        X = np.array(X, dtype='float32')
        if 'val' not in self.dset_dir:
            X = self.augmenter.flow(X, batch_size=len(X), shuffle=False).next()
        y = np.array([self.labels[k] for k in indexes], dtype='uint8')

        return X, to_categorical(y, self.num_classes)

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args, filepath, f_output, model_n, method=None):

    out_path = './../../data/'
    if not os.path.exists(out_path):
        print(f"Generating folder {out_path}")
        os.makedirs(out_path)

    root_dir = out_path + 'imagenette2/'
    params = {'batch_size': args.batch_size, 'image_format': 'JPEG', 'new_shape': 128}

    print(model_n)
    base_model_name = args.model_name
    if args.extension is not None:
        base_model_name = re.sub('_' + args.extension, '', base_model_name)

    # Extracting statistics for every model-set combination and history for learning curves
    history = []
    test_acc = np.zeros(args.repetitions)
    test_loss = np.zeros_like(test_acc)
    training_time = []
    callbacks = []
    agg_cm = []

    if os.path.exists(root_dir + "train/data.npz"):
        npzfile = np.load(root_dir + "train/data.npz", allow_pickle=True)
        x_train = npzfile['arr_0']
        y_train = npzfile['arr_1']
        # class_mapping = npzfile['arr_2']
    else:
        training_generator = ImagenetteGenerator(root_dir=root_dir,
                                                 dset_dir=root_dir + 'train/',
                                                 statistics=[],
                                                 **params)
        x_train, y_train = training_generator.retrieve_set()
        # class_mapping = training_generator.class_mapping

    if os.path.exists(root_dir + "val/data.npz"):
        npzfile = np.load(root_dir + "val/data.npz", allow_pickle=True)
        x_val = npzfile['arr_0']
        y_val = npzfile['arr_1']
    else:
        validation_generator = ImagenetteGenerator(root_dir=root_dir,
                                                 dset_dir=root_dir + 'val/',
                                                 statistics=[],
                                                 res_shape=128,
                                                 **params)
        x_val, y_val = validation_generator.retrieve_set()

    if args.pixel_mean:
        x_train -= np.mean(x_train, axis=0)
        x_val -= np.mean(x_val, axis=0)

    training_generator = ImagenetteGenerator_inmem(x_train, y_train, batch_size=args.batch_size)
    validation_generator = ImagenetteGenerator_inmem(x_val, y_val, batch_size=args.batch_size, val=True)

    for i in range(args.repetitions):

        sched = globals()[args.scheduler]
        if 'stage' in args.scheduler:
            print(args.scheduler)
            cb_decayLR = tf.keras.callbacks.LearningRateScheduler(sched(args.learning_rate, args.num_epochs),
                                                                  verbose=0)
        else:
            cb_decayLR = tf.keras.callbacks.LearningRateScheduler(sched, verbose=0)

        if not callbacks:
            callbacks.append(cb_decayLR)
        else:
            callbacks[0] = cb_decayLR

        confusion_m_cb = ConfusionMatrixCB(validation_generator)
        callbacks.append(confusion_m_cb)

        # Resetting the model for the next iteration
        input_shape = [params['new_shape'], params['new_shape'], 3]
        print('Loading model: ', base_model_name)
        optimizer = tf.keras.optimizers.SGD(args.learning_rate, momentum=0.9, nesterov=True)

        if method is not None:
            model = select_model(input_shape, base_model_name, optimizer, args.weight_decay, method, gpus=args.gpus)
        else:
            model = select_model(input_shape, base_model_name, optimizer, args.weight_decay, gpus=args.gpus)

        start_train = time.time()
        hist = model.fit_generator(generator=training_generator,
                                   validation_data=validation_generator,
                                   epochs=args.num_epochs,
                                   verbose=2, callbacks=callbacks)

        training_time.append(time.time() - start_train)

        test_loss[i], test_acc[i] = model.evaluate_generator(validation_generator, verbose=0)

        history.append(hist.history)

        agg_cm.append(confusion_m_cb.get_cm())
        callbacks = callbacks[:-1]

        if i == args.repetitions - 1:
            model.save(filepath['models'] + filepath['dataset'] + model_n + '.h5')

    # Store history
    with open(filepath['history'] + filepath['dataset'] + 'history_' + model_n + '.txt', 'wb') as f_history:
        pickle.dump(history, f_history)

    mean_agg_cm = np.mean(agg_cm, axis=0)
    std_agg_cm = np.std(agg_cm, axis=0, ddof=1)
    mean_agg_cm = np.round(mean_agg_cm / np.sum(mean_agg_cm, axis=1), 3)

    mean_test_loss = np.mean(test_loss)
    std_test_loss = np.std(test_loss, ddof=1)
    mean_test_acc = np.mean(test_acc)
    std_test_acc = np.std(test_acc, ddof=1)

    # Writing statistics to file
    print("****************************************", file=f_output)
    print("Model: ", model_n, file=f_output)
    print(f"Mean test loss: {mean_test_loss} +- {std_test_loss} ", file=f_output)
    print(f"Mean test accuracy: {mean_test_acc} +- {std_test_acc}\n", file=f_output)

    print("Aggregated confusion matrix: mean +- std", file=f_output)
    print(f"{mean_agg_cm}\n", file=f_output)
    print(f"{std_agg_cm}\n", file=f_output)

    print(f"Mean training time: {np.mean(training_time)} +- {np.std(training_time, ddof=1)}", file=f_output)
    print("****************************************\n\n\n", file=f_output)

    learning_curves(history, model_n=model_n, filepath=filepath['graphs'] + filepath['dataset'])


def main():

    args, filepath, f_output, orig_size = setup()
    train(args, filepath, f_output, model_n=args.model_name)


if __name__ == '__main__':
    main()
