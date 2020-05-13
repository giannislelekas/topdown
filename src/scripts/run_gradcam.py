import os
import numpy as np
import cv2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import to_categorical

from scripts.run_imagenette import ImagenetteGenerator
from scripts.run_BUvsTD import parse_input


'''
Script for running the 3rd experiment, explainality and localization.
For running on MNIST and Fashion-MNIST, 4 checkpoints of models trained from scratch are needed.
To this goal, train NIN_light and NIN_light_TD by setting the argument `save_all_weight=True`, of method train
in main() of run_BUvsTD.py.
'''

'''
Commandline inputs: 

 -d MNIST
    -m NIN_light -b 128 -r 4
    -m NIN_light_TD -b 128 -r 4

 -d FMNIST (Fashion-MNIST)
    -m NIN_light -b 128 -r 4
    -m NIN_light_TD -b 128 -r 4

 -d IMAGENETTE:
    -m ResNet18 -b 128 -r 1 -p True
    -m ResNet18_TD -b 64 -r 1 -p True
    
NOTE: Use -pr True if you want to use model with pretrained weights.
'''

def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    
    # Process CAMs
    (H, W) = images.shape[1:-1]
    new_cams = np.empty((images.shape[0], H, W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()
    
    return new_cams


def run_gradcam_batch(X, model, layer_name, batch_size):

    N = len(X)
    y_pred = model.predict(X, batch_size=batch_size)
    top = np.argmax(y_pred, axis=-1)

    gradcam = np.empty((X.shape[:-1]))
    for i in range((N + batch_size - 1) // batch_size):
        start = i * batch_size
        end = min((i + 1) * batch_size, N)
        gradcam[start:end] = grad_cam_batch(model, X[start:end], top[start:end], layer_name)

    return gradcam, top


def main():

    args = parse_input()
    out_path = './../../'

    if not os.path.exists(out_path + 'output/gradcam/'):
        print(f"Generating folder {out_path + 'output/gradcam/'}")
        os.makedirs(out_path + 'output/gradcam/')

    if args.dataset == 'IMAGENETTE':
        root_dir = out_path + 'data/imagenette2/'
        params = {'batch_size': args.batch_size, 'image_format': 'JPEG', 'res_shape': 128}

        if os.path.exists(root_dir + "val/data.npz"):
            npzfile = np.load(root_dir + "val/data.npz", allow_pickle=True)
            x_test = npzfile['arr_0']
            y_test = npzfile['arr_1']
        else:
            validation_generator = ImagenetteGenerator(root_dir=root_dir,
                                                       dset_dir=root_dir + 'val/',
                                                       statistics=[],
                                                       **params)
            x_test, y_test = validation_generator.retrieve_set()

        m = np.mean(x_test, 0)
        layers = ['act_1', 'act_2', 'act_3', 'act_4', 'act_5']

    else:
        if args.dataset == 'MNIST':
            (x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train[..., np.newaxis]
            x_test = x_test[..., np.newaxis]
        elif args.dataset == 'FMNIST':
            (x_train, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            x_train = x_train[..., np.newaxis]
            x_test = x_test[..., np.newaxis]
        else:
            (x_train, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        m = np.mean(x_train, axis=0)
        layers = ['act_1', 'act_2', 'act_3', 'act_4']

    if not args.pixel_mean:
        # print("P: ", args.pixel_mean)
        m = 0

    if args.pretrained_weights:
        lr = 0.001
        if args.dataset == 'IMAGENETTE' and 'TD' in args.model_name:
            lr = 0.0005

    ind = np.arange(len(x_test), dtype=int)
    np.random.seed(0)
    np.random.shuffle(ind)
    unshuffle_ind = np.argsort(ind)

    gradcam_list = []
    y_pred_list = []
    for r in range(args.repetitions):
        if args.dataset != 'IMAGENETTE':
            if not args.pretrained_weights:
                print(f"Loading model: {args.model_name}_it{r}")
                model = load_model(f'{out_path}output/models/{args.dataset}/{args.model_name}_it{r}.h5')
            else:
                from models.models_BUvsTD import select_model
                print(f"Loading pretrained weights for mdoel: {args.model_name}_it{r}")
                model = select_model(x_test.shape[1:], args.model_name, SGD(lr, momentum=0.9, nesterov=True), 0)
                model.load_weights(f'{out_path}output/trained_weights/{args.dataset}/{args.model_name}_it{r}.h5')
            print("Layers: ", layers)
        else:
            if not args.pretrained_weights:
                print(f"Loading model: {args.model_name}")
                model = load_model(f'{out_path}output/models/{args.dataset}/{args.model_name}.h5')
            else:
                from models.models_imagenette import select_model
                print(f"Loading pretrained weights for mdoel: {args.model_name}")
                model = select_model(x_test.shape[1:], args.model_name, SGD(lr, momentum=0.9, nesterov=True), 1e-3)
                model.load_weights(f'{out_path}output/trained_weights/{args.dataset}/{args.model_name}.h5')

        model.evaluate(x_test[ind] - m, to_categorical(y_test[ind], 10), batch_size=args.batch_size)
        gradcam = {}
        for layer in layers:
            g, y_pred = run_gradcam_batch(x_test[ind] - m, model, layer, args.batch_size)
            gradcam[layer] = g[unshuffle_ind].astype(np.float32)
            print(gradcam[layer].shape, gradcam[layer].dtype)

        gradcam_list.append(gradcam)
        y_pred_list.append(y_pred[unshuffle_ind])

    np.savez(f'{out_path}output/gradcam/{args.model_name}_{args.dataset}_gradcam', gradcam_list, y_pred_list)


if __name__ == '__main__':
    main()


