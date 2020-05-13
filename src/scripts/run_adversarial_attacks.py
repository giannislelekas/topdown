import os
import re

import foolbox
from foolbox.distances import Linf
from foolbox.criteria import TargetClass
import numpy as np

import tensorflow as tf
from sklearn.utils.multiclass import unique_labels
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import to_categorical

from lib.helper_functions import normalization, batch_generator, ind_perturbed, unshuffle_index, distance
from models.models_BUvsTD import select_model
from scripts.run_BUvsTD import parse_input


'''
Script for running the 2nd, Adversarial robustness experiment.
'''

'''
Commandline inputs: 

 -d MNIST
    -m LeNetFC -r 3 -b 128 -ab 128 -sh True (-w 0)
    -m LeNetFC_TD -r 3 -b 128 -ab 128 -w 0 -abl True -sh True
    -m NIN_light -r 3 -b 128 -ab 128 -sh True (-w 0)
    -m NIN_light_TD -r 3 -b 128 -ab 128 -w 0 -abl True -sh True

 -d FMNIST (Fashion-MNIST)
    -m LeNetFC -r 3 -b 128 -ab 128 -sh True (-w 0)
    -m LeNetFC_TD -r 3 -b 128 -ab 128 -w 0 -abl True -sh True
    -m NIN_light -l -r 3 -b 128 -ab 128 -sh True (-w 0)
    -m NIN_light_TD -r 3 -b 128 -ab 128 -w 0 -abl True -sh True

 -d CIFAR10
    -m ResNet -r 3 -b 128 -ab 128 -p True -sh True (-w 5e-4)
    -m ResNet_TD -r 3 -b 128 -ab 128 -p True -w 5e-4 -abl True -sh True
    -m NIN -r 3 -b 128 -ab 128 -p True -sh True (-w 5e-4)
    -m NIN_TD -r 3 -b 128 -ab 128 -p True -w 5e-4 -abl True -sh True

    augmented cases:
    -m ResNet -r 3 -b 128 -ab 128 -p True -ex aug -sh True (-w 1e-4)
    -m ResNet_TD -r 3 -b 128 -ab 128 -p True -ex aug -w 1e-4 -abl True -sh True
    -m NIN -r 3 -b 128 -ab 128 -p True -ex aug -sh True (-w 1e-4)
    -m NIN_TD -r 3 -b 128 -ab 128 -p True -ex aug -w 1e-4 -abl True -sh True

NOTE: The _uni and _rev TD variants share the inputs of the corresponding TD.
      Use -pr True if you want to use model with pretrained weights and specify 
      the weight decay -w given in parenenthesis.
      If there is an extension to the filename with the model checkpoint, e.g.
      NINFC_TD_it0.h5 you need to specify -m NINFC_TD -ex it0.
'''


def setup():

    args = parse_input()
    out_path = './../../output/'
    filepath = {'models': out_path + 'models/',
                'trained_weights': out_path + 'trained_weights/',
                'output': out_path + 'adversarial/output/'}

    if args.extension is not None:
        args.model_name = args.model_name + '_' + args.extension

    filepath['dataset'] = args.dataset + '/'

    directory = filepath['output'] + filepath['dataset']
    if not os.path.exists(directory):
        print(f"Generating folder {directory}")
        os.makedirs(directory)

    return args, filepath


def preprocessing(args):

    if args.dataset == 'MNIST':
        (x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif args.dataset == 'FMNIST':
        (x_train, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (x_train, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 4D datasets for tf
    if len(x_test.shape) == 3:
        x_test = x_test[..., np.newaxis]

    if len(y_test.shape) > 1:
        y_test = np.squeeze(y_test)

    if args.adversarial_batch_size is None:
        args.adversarial_batch_size = y_test.shape[0]

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    mean_x_train = np.mean(x_train, axis=0)
    std_x_train = np.std(x_train, axis=0, ddof=1)

    x_test = normalization(args.normalization_type, train=x_test)

    n_classes = len(unique_labels(y_test))
    y_test_onehot = to_categorical(y_test, num_classes=n_classes)

    return x_test, y_test, y_test_onehot, mean_x_train, std_x_train


def foolbox_attack(name, model):

    target = re.search(r'\d{0,2}$', name).group()
    if target is not None:
        name = re.sub('_' + target, '', name)

    if name == 'FGSM':
        return foolbox.attacks.FGSM(model, distance=Linf)
    elif name == 'S&P':
        return foolbox.attacks.SaltAndPepperNoiseAttack(model, distance=Linf)
    elif name == 'C&W':
        return foolbox.attacks.CarliniWagnerL2Attack(model, distance=Linf)
    elif name == 'SinglePixelAttack':
        return foolbox.attacks.SinglePixelAttack(model, distance=Linf)
    elif name == 'LocalSearchAttack':
        return foolbox.attacks.LocalSearchAttack(model, distance=Linf)
    elif name == 'SpatialAttack':
        return foolbox.attacks.SpatialAttack(model, distance=Linf)
    elif name == 'ShiftsAttack':
        return foolbox.attacks.SpatialAttack(model, distance=Linf)
    elif name == 'BoundaryAttack':
        return foolbox.attacks.BoundaryAttack(model, distance=Linf)
    elif name == 'PointwiseAttack':
        return foolbox.attacks.PointwiseAttack(model, distance=Linf)
    elif name == 'ContrastReductionAttack':
        return foolbox.attacks.ContrastReductionAttack(model, distance=Linf)
    elif name == 'AdditiveUniformNoiseAttack':
        return foolbox.attacks.AdditiveUniformNoiseAttack(model, distance=Linf)
    elif name == 'AdditiveGaussianNoiseAttack':
        return foolbox.attacks.AdditiveGaussianNoiseAttack(model, distance=Linf)
    elif name == 'BlendedUniformNoiseAttack':
        return foolbox.attacks.BlendedUniformNoiseAttack(model, distance=Linf)
    elif name == 'GaussianBlurAttack':
        return foolbox.attacks.GaussianBlurAttack(model, distance=Linf)
    elif name == 'DeepFoolAttack':
        return foolbox.attacks.DeepFoolAttack(model, distance=Linf)
    elif name == 'GenAttack':
        return foolbox.attacks.GenAttack(model, criterion=TargetClass(int(target)), distance=Linf)
    elif name == 'PrecomputedAdversarialsAttack':
        return foolbox.attacks.PrecomputedAdversarialsAttack(model, distance=Linf)
    elif name == 'InversionAttack':
        return foolbox.attacks.InversionAttack(model, distance=Linf)
    elif name == 'HopSkipJumpAttack':
        return foolbox.attacks.HopSkipJumpAttack(model, distance=Linf)
    elif name == 'RandomPGD':
        return foolbox.attacks.RandomPGD(model, distance=Linf)
    else:
        print('Oops')


def run_attacks(args, filepath, x_test, y_test, y_test_onehot, attacks, m_train, save_all=True):

    base_model_name = args.model_name
    if args.extension is not None:
        base_model_name = re.sub('_' + args.extension, '', base_model_name)

    if not args.pretrained_weights:
        model = load_model(filepath['models'] + filepath['dataset'] + args.model_name + '.h5')
    else:
        model = select_model(x_test.shape[1:], base_model_name, SGD(0.001, momentum=0.9, nesterov=True), args.weight_decay)
        print(f"Loading pretrained weights for model: {base_model_name}")
        model.load_weights(filepath['trained_weights'] + filepath['dataset'] + args.model_name + '.h5')

    if args.pixel_mean:
        fb_model = foolbox.models.TensorFlowModel.from_keras(model=model, bounds=(0., 1.), preprocessing={'mean': m_train})
    else:
        m_train = 0
        fb_model = foolbox.models.TensorFlowModel.from_keras(model=model, bounds=(0., 1.))

    print("Model: ", args.model_name)
    test_loss, test_acc = model.evaluate(x_test - m_train, y_test_onehot, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(model.predict(x_test - m_train, batch_size=args.batch_size), axis=-1)
    or_acc = np.sum(y_test == y_pred)
    print(f"Test loss: {test_loss}, test acc: {test_acc}")
    # print(f"Or acc: ", or_acc)

    # For extracting statistics. Initialising with empty lists
    if os.path.exists(filepath['output'] + filepath['dataset'] + args.model_name + '.npz'):
        print(f"Resuming attacks for model: {args.model_name}")
        npzfile = np.load(filepath['output'] + filepath['dataset'] + args.model_name + '.npz', allow_pickle=True)
        labels = npzfile['arr_0'].item()
        perturbed = npzfile['arr_1'].item()
        distances = npzfile['arr_2'].item()
        l2_distances = npzfile['arr_3'].item()
        sorted_distances = npzfile['arr_4'].item()
        adv_acc = npzfile['arr_5'].item()
        dict_loss = npzfile['arr_9'].item()
        dict_acc = npzfile['arr_10'].item()
    else:
        print(f"Starting attack from scratch for model: {args.model_name}")
        labels = {}
        perturbed = {}
        distances = {}
        l2_distances = {}
        sorted_distances = {}
        adv_acc = {}
        dict_loss = {}
        dict_acc = {}

    for r in range(args.repetitions):

        batch_ind = batch_generator(len(y_test), batch_size=args.adversarial_batch_size, shuffle=args.shuffle)
        # print("B: ", batch_ind)

        for a in attacks:

            label = []
            dist = []
            pert = []

            fb_attack = foolbox_attack(a, fb_model)

            for b in batch_ind:
                if a == 'ShiftsAttack':
                    adversarials = fb_attack(x_test[b], y_test[b], unpack=False, do_rotations=False)
                else:
                    adversarials = fb_attack(x_test[b], y_test[b], unpack=False)

                label.extend([ad.adversarial_class for ad in adversarials])
                dist.extend([ad.distance.value for ad in adversarials])
                pert.extend([ad.perturbed for ad in adversarials])
                # print("Label shape: ", len(label))

            label = np.array(label)
            dist = np.array(dist)
            pert = np.array(pert)
            # print(f"label: {label.shape}, dist: {dist.shape}, pert: {pert.shape}")

            if args.shuffle:
                unshuffle_ind = unshuffle_index(batch_ind)
                label = label[unshuffle_ind]
                dist = dist[unshuffle_ind]
                pert = pert[unshuffle_ind]

            ind_pert = ind_perturbed(dist)
            ind_not_inf = ind_not_infinite(dist)

            dist = dist.astype(np.float32)
            if label.dtype == 'object':
                label = convert_object_array(label, in_shape=len(x_test), ind_pert=ind_not_inf, dtype=np.int8)
                pert = convert_object_array(pert, in_shape=x_test.shape[1:], ind_pert=ind_not_inf, dtype=np.float32)

            x_adv = np.array(x_test, copy=True)
            if len(ind_pert) > 0:
                x_adv[ind_pert] = pert[ind_pert]
            l2_dist = distance(x_test, x_adv, 2)

            if a not in labels.keys():
                labels[a] = label
                distances[a] = dist
                perturbed[a] = pert
                l2_distances[a] = l2_dist
            else:
                ind = np.where(np.logical_and(l2_dist > 0, np.logical_or(l2_dist < l2_distances[a], l2_distances[a] == 0)))[0]

                if np.sum(ind) > 0:
                    labels[a][ind] = label[ind]
                    distances[a][ind] = dist[ind]
                    # print("Pert: ", pert[ind].shape)
                    # print("PP: ", perturbed[a][ind].shape)
                    perturbed[a][ind] = pert[ind]
                    l2_distances[a][ind] = l2_dist[ind]

            if r == args.repetitions-1:
                sorted_dist, acc = evaluate(x_test, distances[a], perturbed[a], or_acc)
                sorted_distances[a] = sorted_dist
                adv_acc[a] = acc

    if args.ablation:
        dict_loss, dict_acc = ablation(args, model, x_test-m_train, y_test_onehot, distances, perturbed, m_train)

    sorted_distances, adv_acc = to_nparray(sorted_distances), to_nparray(adv_acc)
    SR, confidence, ptb, ptbr = attack_statistics(args, model, x_test, distances, perturbed, or_acc, m_train)

    if save_all:
        np.savez(filepath['output'] + filepath['dataset'] + args.model_name,
                 labels, perturbed, distances, l2_distances, sorted_distances, adv_acc,
                 SR, ptb, ptbr, dict_loss, dict_acc)

    return labels, perturbed, distances, l2_distances, sorted_distances, adv_acc, SR, ptb, ptbr, dict_loss, dict_acc


def convert_object_array(array, in_shape, ind_pert, value_None=-1.0, dtype=int):

    n = len(array)

    if isinstance(in_shape, int):
        np_array = value_None * np.ones(n)
        if len(ind_pert) > 0:
            np_array[ind_pert] = array[ind_pert]
    else:
        np_array = value_None * np.ones((n, in_shape[0], in_shape[1], in_shape[2]))
        if len(ind_pert) > 0:
            x_adv = []
            for i in ind_pert:
                x_adv.append(array[i])
            x_adv = np.array(x_adv)
            np_array[ind_pert] = x_adv

    return np_array.astype(dtype)


def to_nparray(d):

    keys = list(d.keys())

    for key in keys:
        d[key] = np.array(d[key])

    return d


def extract_perturbed(distances, perturbed):

    ind = ind_perturbed(distances)
    x_adv = []
    for i in ind:
        x_adv.append(perturbed[i])

    return np.array(x_adv)


def ind_not_infinite(distances):

    return np.where(distances != np.inf)[0]


def evaluate(x_test, distances, perturbed, or_acc):

    ind = ind_perturbed(distances)
    x_adv = perturbed[ind]
    l2_dist = np.zeros(len(distances))
    l2_dist[ind] = distance(x_test[ind], x_adv, 2)

    ind_non_pert = np.where(distances == np.inf)[0]
    l2_dist[ind_non_pert] = np.inf

    unique_dist, counts = np.unique(l2_dist, return_counts=True)
    sorted_dist = np.zeros_like(l2_dist)
    acc = or_acc * np.ones_like(sorted_dist)
    tot_c = 0
    for i in range(len(unique_dist)):
        sorted_dist[tot_c:tot_c + counts[i]] = unique_dist[i]
        if unique_dist[i] == np.inf:
            acc[tot_c: tot_c + counts[i]] = acc[max(tot_c - 1, 0)]
        elif unique_dist[i] > 0:
            acc[tot_c: tot_c + counts[i]] = acc[max(tot_c - 1, 0)] - (np.arange(counts[i]) + 1)
        tot_c += counts[i]

    return sorted_dist, acc/len(x_test)


def attack_statistics(args, model, x_test, distances, perturbed, or_acc, m_train):

    attacks = list(distances.keys())

    SR = {}
    ptb = {}
    ptbr = {}
    confidence = {}

    for a in attacks:
        ind = ind_perturbed(distances[a])

        SR[a] = np.round(100 * np.sum(np.logical_and(np.greater(distances[a], 0), ~np.isinf(distances[a])))/or_acc, 2)

        if len(ind) > 0:
            x_adv = perturbed[a][ind]
            confidence[a] = np.max(model.predict(x_test - m_train, batch_size=args.batch_size), axis=-1)
            ptb[a] = distance(x_test[ind], x_adv, 'ptb')
            ptbr[a] = distance(x_test[ind], x_adv, 'ptbr')

    return SR, confidence, ptb, ptbr


'''
Reintroducing extracted perturbations to a single input. The other inputs are fed with the original inputs.
NOTE: Due to some adversarial samples lying close to the decision boundary, it is possible that sometimes when
re-evaluated are no longer misclassified.
'''
def ablation(args, model, x_test, y_test_onehot, distances, perturbed, m_train):

    attacks = list(distances.keys())
    loss = np.zeros((len(attacks), 9))
    acc = np.zeros_like(loss)

    dict_loss = {}
    dict_acc = {}

    base_model_name = args.model_name
    if args.extension is not None:
        base_model_name = re.sub('_' + args.extension, '', base_model_name)
    base_model_name += '_multi'

    print(base_model_name)

    multi_model = select_model(x_test.shape[1:], base_model_name, optimizer=SGD(0.001, momentum=0.9, nesterov=True),
                               weight_decay=args.weight_decay)
    multi_model.set_weights(model.get_weights())

    for a, i in zip(attacks, range(len(attacks))):

        ind = ind_perturbed(distances[a])

        x_adv = np.array(x_test, copy=True)
        if ind.shape[0] > 0:
            x_adv[ind] = perturbed[a][ind] - m_train

        loss[i, 0], acc[i, 0] = model.evaluate(x_test, y_test_onehot, batch_size=args.batch_size, verbose=0)
        loss[i, 1], acc[i, 1] = multi_model.evaluate([x_test, x_test, x_test], y_test_onehot,
                                                     batch_size=args.batch_size, verbose=0)

        loss[i, 2], acc[i, 2] = multi_model.evaluate([x_adv, x_test, x_test], y_test_onehot,
                                                     batch_size=args.batch_size, verbose=0)
        loss[i, 3], acc[i, 3] = multi_model.evaluate([x_test, x_adv, x_test], y_test_onehot,
                                                     batch_size=args.batch_size, verbose=0)
        loss[i, 4], acc[i, 4] = multi_model.evaluate([x_test, x_test, x_adv], y_test_onehot,
                                                     batch_size=args.batch_size, verbose=0)

        loss[i, 5], acc[i, 5] = multi_model.evaluate([x_adv, x_adv, x_test], y_test_onehot,
                                                     batch_size=args.batch_size, verbose=0)
        loss[i, 6], acc[i, 6] = multi_model.evaluate([x_adv, x_test, x_adv], y_test_onehot,
                                                     batch_size=args.batch_size, verbose=0)
        loss[i, 7], acc[i, 7] = multi_model.evaluate([x_test, x_adv, x_adv], y_test_onehot,
                                                     batch_size=args.batch_size, verbose=0)
        loss[i, 8], acc[i, 8] = multi_model.evaluate([x_adv, x_adv, x_adv], y_test_onehot,
                                                     batch_size=args.batch_size,
                                                     verbose=0)

        dict_loss[a] = loss[i]
        dict_acc[a] = acc[i]

    return dict_loss, dict_acc


def main():

    # K.set_learning_phase(0)

    args, filepath = setup()

    # print("Shuffle:", args.shuffle)
    attacks = ['SinglePixelAttack', 'SpatialAttack', 'ShiftsAttack', 'PointwiseAttack',
               'GaussianBlurAttack', 'ContrastReductionAttack', 'AdditiveUniformNoiseAttack',
               'AdditiveGaussianNoiseAttack', 'S&P', 'BlendedUniformNoiseAttack']

    print('Running attacks')
    x_test, y_test, y_test_onehot, m_train, s_train = preprocessing(args)
    run_attacks(args, filepath, x_test, y_test, y_test_onehot, attacks, m_train)

    print('Completed')


if __name__ == '__main__':
    main()

