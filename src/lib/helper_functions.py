import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

'''
Computing the distance of certain order between two inputs; dimensions of the inputs should match.
'''
def distance(x, y, order):
    axis = tuple(np.arange(1, len(x.shape), dtype=int))
    input_shape = x.shape

    if order == 0:
        return np.sum((x - y) != 0, axis)
    elif order == 1:
        return np.sum(np.abs(x - y), axis)
    elif order == 2:
        return np.sqrt(np.sum(np.square(x - y), axis))
    elif order == 'inf':
        return np.max(np.abs(x - y), axis)
    elif order == 'mse':
        return np.mean(np.sum(np.square(x - y), axis))
    elif order == 'ptb':
        l1 = distance(x, y, 1)

        return np.round((1 / x.shape[0]) * (1 / np.prod(input_shape[1:])) * np.sum(l1), 4)
    elif order == 'ptbr':
        l0 = distance(x, y, 0)

        return np.round(100 * (1 / x.shape[0]) * (1 / np.prod(input_shape[1:])) * np.sum(l0), 2)
    else:
        print('Give correct order')


'''
Function for extracting the indices of perturbed samples.
'''
def ind_perturbed(adversarial_distance):
    ind = np.where(np.logical_and(np.greater(adversarial_distance, 0), np.less_equal(adversarial_distance, 1)))[0]

    return ind

'''
Function for extracting the indices of samples that no perturbation was found.
'''
def ind_non_perturbable(adversarial_distance):
    ind = np.where(np.isinf(adversarial_distance))[0]

    return ind


'''
Custom batch generator.
'''
def batch_generator(length, batch_size, shuffle=False):

    if isinstance(batch_size, float):
        batch_size = int(np.ceil(length*batch_size))
    else:
        if batch_size > length:
            batch_size = length

    ind = np.arange(length, dtype=int)
    n = int(np.ceil(length / batch_size))
    if shuffle:
        np.random.shuffle(ind)

    if n == 1:
        return [ind.tolist()]
    batch_ind = np.reshape(ind[:(n-1)*batch_size], [n-1, batch_size]).tolist()
    batch_ind.append(ind[(n-1)*batch_size:].tolist())

    return batch_ind


'''
Function for reversing a shuffling operation.
'''
def unshuffle_index(batch_ind):

    batch_ind_flat = np.array([x for sublist in batch_ind for x in sublist])
    return np.argsort(batch_ind_flat)


'''
Function for scaling the input, so all features are within [0, 1].
'''
def min_max_scaling(train, test=None):
    mms = MinMaxScaler()

    shape_train = train.shape
    train = np.reshape(train, [shape_train[0], np.prod(shape_train[1:])])
    mms.fit(train)
    train = mms.transform(train)
    train = np.reshape(train, shape_train)

    if test is not None:
        shape_test = test.shape
        test = np.reshape(test, [shape_test[0], np.prod(shape_test[1:])])
        test = mms.transform(test)
        test = np.reshape(test, shape_test)

        return train, test

    return train


'''
Simple scaling, dividing by the max.
'''
def max_scaling(train, test=None):
    if test is not None:
        return train / np.max(train), test / np.max(test)

    return train / np.max(train)


'''
Function for scaling the input, so all features are of 0 mean and unit variance.
'''
def zero_mean_unit_variance(train, test=None, use_std=True):
    train_mean = np.mean(train, axis=0)
    if test is not None:
        test_mean = np.mean(test, axis=0)
    if use_std:
        train_std = np.std(train, axis=0)
        if test is not None:
            test_std = np.std(test, axis=0)
    else:
        train_std = np.ones_like(train_mean)
        if test is not None:
            test_std = np.ones_like(test_mean)
    train = (train - train_mean) / (train_std + 1e-32)

    if test is not None:
        test = (test - test_mean) / (test_std + 1e-32)

        return train, test

    return train


'''
Function for scaling the input, so input is positive and within [0, 1].
'''
def translate_pos(train):
    train_pos = np.array(train, copy=True)
    orig_shape = train_pos.shape
    train_pos = np.moveaxis(train_pos, -1, 1)
    perm_shape = train_pos.shape
    train_pos = np.reshape(train_pos, [perm_shape[0], perm_shape[1], np.prod(perm_shape[2:])])
    min_per_image = np.min(train_pos, axis=2)
    train_pos[min_per_image < 0] = train_pos[min_per_image < 0] - np.tile(
        min_per_image[min_per_image < 0][..., np.newaxis], (1, np.prod(perm_shape[2:])))
    train_pos = train_pos / (np.tile(np.max(train_pos, axis=2)[..., np.newaxis], (1, np.prod(perm_shape[2:]))) + 1e-32)
    train_pos = np.moveaxis(train_pos, 1, -1)
    train_pos = np.reshape(train_pos, orig_shape)

    return train_pos

'''
Subtracting the pixel mean.
'''
def pixel_mean(train, test):

    train_mean = np.mean(train, 0)

    return train - train_mean, test - train_mean


'''
Wrapper function for all scaling functions
'''
def normalization(norm_type, train, test=None):
    if norm_type != 'no':

        if norm_type == 'max':
            if test is not None:
                train, test = max_scaling(train, test)
            else:
                train = max_scaling(train)
        elif norm_type == 'min_max':
            if test is not None:
                train, test = min_max_scaling(train, test)
            else:
                train = min_max_scaling(train)
        elif norm_type == 't_pos':
            train = translate_pos(train)
            if test is not None:
                test = translate_pos(test)
        elif norm_type == 'zero_unit':
            if test is not None:
                train, test = zero_mean_unit_variance(train, test, use_std=True)
            else:
                train = zero_mean_unit_variance(train, use_std=True)
        else:
            raise KeyError("Give correct type of norm_type, among {max, min_max and zero_unit}")

    if test is not None:
        return train, test
    else:
        return train


'''
Function for extracting training and validation splits.
'''
def validation_split(x_train, y_train, val_split, stratified=True):

    if stratified:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split)
        for train_ind, val_ind in sss.split(x_train, y_train):
            return train_ind, val_ind
    else:
        n = len(x_train)
        ind = np.arange(n)
        val_ind = np.random.choice(n, int(np.ceil(n*val_split)), replace=False)
        train_ind = np.isin(ind, val_ind, invert=True)

        return train_ind, val_ind


