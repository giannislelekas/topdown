import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models
from sklearn.metrics import confusion_matrix

# For validation on multiple scales
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import Sequence


'''
Callback for training with multiple validatons sets.
'''
class MultiVal(tf.keras.callbacks.Callback):

    def __init__(self, val_data, val_target, batch_size):

        super().__init__()
        self.validation_data = val_data
        self.validation_target = val_target
        self.batch_size = batch_size

        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):

        for validation_data in self.validation_data:
            loss, acc = self.model.evaluate(validation_data, y=self.validation_target, batch_size=self.batch_size, verbose=0)
            self.val_loss.append(loss)
            self.val_acc.append(acc)

    def to_dict(self):

        return {'val_loss': self.val_loss, 'val_acc': self.val_acc}


'''
Callback for assessing the contrubutinon of each scale to the merging.
'''
class TrackAddition(tf.keras.callbacks.Callback):

    def __init__(self, val_data, num, batch_size, num_contrib):

        super().__init__()
        self.val_data = val_data
        self.num = num
        self.batch_size = batch_size
        self.num_contrib = num_contrib
        if self.num < len(self.val_data):
            self.ind = np.random.choice(len(self.val_data), self.num, replace=False)
            self.track_data = self.val_data[self.ind]
        else:
            self.ind = np.arange(len(self.val_data))
            self.track_data = self.val_data
        if self.num == 1:
            self.track_data = self.track_data[..., np.newaxis]

        # self.mean_contrib_1 = []
        # self.std_contrib_1 = []
        # self.mean_contrib_2 = []
        # self.std_contrib_2 = []
        self.mean_c = {}
        self.std_c = {}

        self.output = []
        self.add_layers = []
        self.ind_add = []
        self.track_model = []

    def set_data(self, ind, val_data):

        self.ind = ind
        self.val_data = val_data

    def layers_ind(self):

        if not self.add_layers:
            self.add_layers = [layer.name for layer in self.model.layers if 'add' in layer.name][1:]
            layers = [layer.name for layer in self.model.layers]
            ind = np.where(np.isin(layers, self.add_layers))[0]

            c = 0
            for i in ind:
                self.ind_add.append(np.arange(i-self.num_contrib, i+1, dtype='int'))
                self.mean_c[f'c_{c}'] = []
                self.std_c[f'c_{c}'] = []
                c += 1
            self.ind_add = [x for sublist in self.ind_add for x in sublist]

        self.output = []
        for i in self.ind_add:
            self.output.append(self.model.get_layer(index=i).output)
        print(self.output)

        self.track_model = models.Model(inputs=self.model.input, outputs=self.output)

    def on_train_begin(self, logs=None):

        self.layers_ind()

    def on_epoch_end(self, epoch, logs=None):

        output = self.track_model.predict(self.track_data, batch_size=self.batch_size, verbose=0)

        n = len(self.add_layers)
        s = len(output) // n

        for i in range(n):
            total = np.sum(output[i*s + s-1], axis=(1, 2, 3))
            m_c = []
            s_c = []
            for j in range(i*s, (i+1)*s-1):
                m_c.append(np.mean(np.sum(output[j], axis=(1, 2, 3))/total, axis=0))
                s_c.append(np.std(np.sum(output[j], axis=(1, 2, 3))/total, axis=0, ddof=1))
            self.mean_c[f'c_{i}'].append(m_c)
            self.std_c[f'c_{i}'].append(s_c)

    def get_contrib(self):

        return [self.mean_c, self.std_c]


'''
Callback for assessing convergence error of low and high components of the target function.
'''
class FilterError(tf.keras.callbacks.Callback):

    def __init__(self, x_test, y_test, batch_size, delta, epochs, num_ckpts, num_samples=0.1):

        if isinstance(num_samples, float):
            self.num_samples = int(np.ceil(num_samples*len(y_test)))
        else:
            self.num_samples = num_samples

        print(self.num_samples)
        ind = np.random.choice(len(y_test), self.num_samples, replace=False)
        self.x_test = x_test[ind]
        self.y_test = y_test[ind]
        self.batch_size = batch_size
        self.delta = delta
        self.epochs = epochs
        self.num_ckpts = num_ckpts

        self.ckpts = np.ceil(np.linspace(0, self.epochs-1, self.num_ckpts)).astype(int)
        print(self.ckpts)
        self.e_low = []
        self.e_high = []

    def error(self, y_pred):

        G = np.array([x - self.x_test for x in self.x_test])
        G = np.exp(-np.sum(np.square(G), axis=(2, 3, 4)) / (2 * self.delta))
        G = G[..., np.newaxis]

        y_low = np.sum(self.y_test[np.newaxis, ...] * G, axis=1) / np.sum(G, axis=1)
        h_low = np.sum(y_pred[np.newaxis, ...] * G, axis=1) / np.sum(G, axis=1)

        # print(G.shape, y_low.shape, h_low.shape)

        y_high = self.y_test - y_low
        h_high = y_pred - h_low

        e_low = np.sqrt(np.sum(np.sum(np.power(y_low - h_low, 2), axis=1)) / np.sum(np.sum(np.power(y_low, 2), axis=1)))
        e_high = np.sqrt(
            np.sum(np.sum(np.power(y_high - h_high, 2), axis=1)) / np.sum(np.sum(np.power(y_high, 2), axis=1)))

        return e_low, e_high

    def error_loop(self, y_pred):

        y_low = np.zeros_like(self.y_test)
        h_low = np.zeros_like(self.y_test)
        for i in range(len(self.y_test)):

            G = np.exp(-np.sum(np.square(self.x_test[i] - self.x_test), axis=(1, 2, 3))/(2*self.delta))[..., np.newaxis]
            y_low[i] = np.sum(self.y_test * G, axis=0) / np.sum(G)
            h_low[i] = np.sum(y_pred * G, axis=0) / np.sum(G)

        y_high = self.y_test - y_low
        h_high = y_pred - h_low
        e_low = np.sqrt(np.sum(np.sum(np.square(y_low - h_low), axis=1)) / np.sum(np.sum(np.square(y_low), axis=1)))
        e_high = np.sqrt(np.sum(np.sum(np.square(y_high - h_high), axis=1)) / np.sum(np.sum(np.square(y_high), axis=1)))

        return e_low, e_high

    def on_epoch_end(self, epoch, logs=None):

        if any(epoch == self.ckpts):

            y_pred = self.model.predict(self.x_test, batch_size=self.batch_size)
            # e_low, e_high = self.error(y_pred)
            e_low, e_high = self.error_loop(y_pred)

            self.e_low.append(e_low)
            self.e_high.append(e_high)

    def get_error(self):

        num_reps = len(self.e_low) // self.num_ckpts

        return np.reshape(self.e_low, [num_reps, self.num_ckpts]), np.reshape(self.e_high, [num_reps, self.num_ckpts])


'''
Callback for extracting a confusion matrix
'''
class ConfusionMatrixCB(tf.keras.callbacks.Callback):

    def __init__(self, validation_generator):

        self.validation_generator = validation_generator

    def on_train_end(self, logs=None):

        # y_true = self.validation_generator.labels[self.validation_generator.indexes]
        y_true = self.validation_generator.y[self.validation_generator.indexes]
        y_pred = np.argmax(self.model.predict_generator(self.validation_generator, verbose=0), axis=-1)
        self.cm = confusion_matrix(y_true, y_pred)

    def get_cm(self):

        return self.cm


'''
Callback for removing skip-connections from a Top-Down model; removing the skip connections removes
a certail scale from the Top-Down model.
'''
class NegateSkipConnections(tf.keras.callbacks.Callback):

    def __init__(self, test_data, layer_names):

        self.test_data = test_data
        self.layer_names = layer_names

        self.test_acc = []

    def get_layers(self):

        self.skip_2 = [layer for layer in self.model.layers if layer.name == self.layer_names[0]][0]
        self.orig_weights_2 = self.skip_2.get_weights()
        self.weights_shape_2 = self.orig_weights_2[0].shape

        self.skip_1 = [layer for layer in self.model.layers if layer.name == self.layer_names[1]][0]
        self.orig_weights_1 = self.skip_1.get_weights()
        self.weights_shape_1 = self.orig_weights_1[0].shape

        print(self.weights_shape_2, self.weights_shape_1)

    def on_train_end(self, logs=None):

        self.get_layers()
        zeros_2 = np.zeros(self.weights_shape_2)
        zeros_2 = [zeros_2, zeros_2, zeros_2, np.ones_like(zeros_2)]
        zeros_1 = np.zeros(self.weights_shape_1)
        zeros_1 = [zeros_1, zeros_1, zeros_1, np.ones_like(zeros_1)]

        # Original
        _, acc = self.model.evaluate(self.test_data, verbose=0)
        self.test_acc.append(acc)

        # Setting mid skip to 0
        self.skip_2.set_weights(zeros_2)
        _, acc = self.model.evaluate(self.test_data, verbose=0)
        self.test_acc.append(acc)
        self.skip_2.set_weights(self.orig_weights_2)

        # Setting high skip to 0
        self.skip_1.set_weights(zeros_1)
        _, acc = self.model.evaluate(self.test_data, verbose=0)
        self.test_acc.append(acc)

        # Setting both to 0
        self.skip_2.set_weights(zeros_2)
        _, acc = self.model.evaluate(self.test_data, verbose=0)
        self.test_acc.append(acc)

        # Restore
        self.skip_1.set_weights(self.orig_weights_1)
        self.skip_2.set_weights(self.orig_weights_2)
        loss, acc = self.model.evaluate(self.test_data, verbose=0)
        print(f"After restore loss:{loss}, acc:{acc}")

    def get_test_acc(self):

        return self.test_acc


'''
A custom augmentation generator.
'''
class AugmentationGenerator(Sequence):

    def __init__(self, X, y, batch_size, shuffle=True):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = np.arange(len(X), dtype=int)
        self.checked = []

        self.augmenter = ImageDataGenerator(
                    featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                    samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0,
                    width_shift_range=0.1, height_shift_range=0.1, brightness_range=None, shear_range=0.0, zoom_range=0,
                    channel_shift_range=0., fill_mode='nearest', cval=0., horizontal_flip=True, vertical_flip=False,
                    rescale=None, preprocessing_function=None, data_format='channels_last', validation_split=0,
                    dtype='float32')
        self.augmenter.fit(X)

        self.on_epoch_end()

    def __len__(self):

        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        self.checked.extend(indexes)

        return self.augmenter.flow(self.X[indexes], batch_size=len(indexes), shuffle=False).next(), self.y[indexes]

    def on_epoch_end(self):

        if len(np.unique(self.checked)) == len(self.X):
            print("All checked")
        self.checked = []

        if self.shuffle:
            np.random.shuffle(self.index)


'''
Callback for a 3-stage learning rate decay scheme.
'''
class scheduler_3_stage(tf.keras.callbacks.Callback):

    def __init__(self, lr, num_epochs):
        self.lr = lr
        self.num_epochs = num_epochs

    def __call__(self, epoch):

        if epoch == int(np.ceil(0.5*self.num_epochs)) or epoch == int(np.ceil(0.8*self.num_epochs)):
            self.lr *= 0.1

        return self.lr


'''
Callback for a 4-stage learning rate decay scheme.
'''
class scheduler_4_stage(tf.keras.callbacks.Callback):

    def __init__(self, lr, num_epochs):
        self.lr = lr
        self.num_epochs = num_epochs

    def __call__(self, epoch):

        if epoch == int(np.ceil(0.4*self.num_epochs)) or epoch == int(np.ceil(0.7*self.num_epochs)) \
                or epoch == int(np.ceil(0.9*self.num_epochs)):
            self.lr *= 0.2

        return self.lr

