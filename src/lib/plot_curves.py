import numpy as np
from matplotlib import pyplot as plt


'''
Function for plotting training and validation curves.
'''
def learning_curves(history, multival=None, model_n=None, filepath=None, plot_from_epoch=0, plot_to_epoch=None):

    n = len(history)
    num_epochs = len(history[0]['loss'])
    if plot_to_epoch is None:
        plot_to_epoch = num_epochs

    IoU_in_history = 'mIoU' in history[0].keys()

    train_loss = np.zeros((n, num_epochs))
    train_acc = np.zeros_like(train_loss)
    val_loss = np.zeros_like(train_loss)
    val_acc = np.zeros_like(train_loss)
    if IoU_in_history:
        train_mIoU = np.zeros_like(train_loss)
        val_mIoU = np.zeros_like(train_loss)

    # For the additional validation curves, which are at different scales
    if multival is not None:
        m = len(multival[0]['val_loss']) // num_epochs
        val_loss_scales = np.zeros((n, num_epochs*m))
        val_acc_scales = np.zeros_like(val_loss_scales)

    for i in range(len(history)):
        train_loss[i, :] = history[i]['loss']
        train_acc[i, :] = history[i]['acc']
        val_loss[i, :] = history[i]['val_loss']
        val_acc[i, :] = history[i]['val_acc']

        if IoU_in_history:
            train_mIoU[i, :] = history[i]['mIoU']
            val_mIoU[i, :] = history[i]['val_mIoU']

        if multival is not None:
            val_loss_scales[i, :] = multival[i]['val_loss']
            val_acc_scales[i, :] = multival[i]['val_acc']

    # Extracting mean values and standard deviations
    mean_train_loss = np.mean(train_loss, 0)
    std_train_loss = np.std(train_loss, 0, ddof=1)
    mean_train_acc = np.mean(train_acc, 0)
    std_train_acc = np.std(train_acc, 0, ddof=1)

    mean_val_loss = np.mean(val_loss, 0)
    std_val_loss = np.std(val_loss, 0, ddof=1)
    mean_val_acc = np.mean(val_acc, 0)
    std_val_acc = np.std(val_acc, 0, ddof=1)

    if IoU_in_history:
        mean_train_mIoU = np.mean(train_mIoU, 0)
        std_train_mIoU = np.std(train_mIoU, 0, ddof=1)
        mean_val_mIoU = np.mean(val_mIoU, 0)
        std_val_mIoU = np.std(val_mIoU, 0, ddof=1)

    # Third dimension corresponds to scales
    if multival is not None:
        val_loss_scales = np.reshape(val_loss_scales, [n, num_epochs, m])
        val_acc_scales = np.reshape(val_acc_scales, [n, num_epochs, m])
        mean_val_loss_scales, mean_val_acc_scales = np.mean(val_loss_scales, axis=0), np.mean(val_acc_scales, axis=0)
        std_val_loss_scales, std_val_acc_scales = np.std(val_loss_scales, axis=0, ddof=1), np.std(val_acc_scales, axis=0, ddof=1)

    # if n == 0:
    #     std_train_loss = 0
    #     std_train_acc = 0
    #     std_val_loss = 0
    #     std_val_acc = 0
    #     std_val_loss_scales = 0
    #     std_val_acc_scales = 0
    # else:
    #     std_train_loss = np.std(train_loss, 0, ddof=1)
    #     std_train_acc = np.std(train_acc, 0, ddof=1)
    #     std_val_loss = np.std(val_loss, 0, ddof=1)
    #     std_val_acc = np.std(val_acc, 0, ddof=1)
    #     std_val_loss_scales = np.std(val_loss_scales, axis=0, ddof=1)
    #     std_val_acc_scales = np.std(val_acc_scales, axis=0, ddof=1)

    if filepath is not None:
        plt.ioff()

    # Plotting mean Loss curves with stds
    plt.figure(figsize=(10, 10))
    plt.title(model_n + '_loss')

    plt.plot(range(plot_to_epoch - plot_from_epoch), mean_train_loss[plot_from_epoch:plot_to_epoch], color="g", label='training')
    plt.fill_between(np.arange(plot_to_epoch - plot_from_epoch), mean_train_loss[plot_from_epoch:plot_to_epoch] - std_train_loss[plot_from_epoch:plot_to_epoch],
                     mean_train_loss[plot_from_epoch:plot_to_epoch] + std_train_loss[plot_from_epoch:plot_to_epoch], alpha=0.2, color="g")

    plt.plot(range(plot_to_epoch - plot_from_epoch), mean_val_loss[plot_from_epoch:plot_to_epoch], color='r', label='validation')
    plt.fill_between(range(plot_to_epoch - plot_from_epoch), mean_val_loss[plot_from_epoch:plot_to_epoch] - std_val_loss[plot_from_epoch:plot_to_epoch],
                     mean_val_loss[plot_from_epoch:plot_to_epoch] + std_val_loss[plot_from_epoch:plot_to_epoch], alpha=0.2, color='r')

    if multival is not None:
        for i in range(m):
            plt.plot(range(plot_to_epoch - plot_from_epoch), mean_val_loss_scales[plot_from_epoch:plot_to_epoch, i], label=f'validation_{i+1}')
            plt.fill_between(range(plot_to_epoch - plot_from_epoch), mean_val_loss_scales[plot_from_epoch:plot_to_epoch, i] - std_val_loss_scales[plot_from_epoch:plot_to_epoch, i],
                             mean_val_loss_scales[plot_from_epoch:plot_to_epoch, i] + std_val_loss_scales[plot_from_epoch:plot_to_epoch, i], alpha=0.2, )

    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend()
    if filepath is not None:
        plt.savefig(filepath + model_n + '_loss' + '.png')


    # Plotting mean Accuracy curves with stds
    plt.figure(figsize=(10, 10))
    plt.title(model_n + '_acc')

    plt.ylim(0, 1.05)
    plt.plot(range(plot_to_epoch - plot_from_epoch), mean_train_acc[plot_from_epoch:plot_to_epoch], color="g", label='training')
    plt.fill_between(np.arange(plot_to_epoch - plot_from_epoch), np.maximum(0, mean_train_acc[plot_from_epoch:plot_to_epoch] - std_train_acc[plot_from_epoch:plot_to_epoch]),
                     np.minimum(1, mean_train_acc[plot_from_epoch:plot_to_epoch] + std_train_acc[plot_from_epoch:plot_to_epoch]), alpha=0.2, color="g")

    plt.plot(range(plot_to_epoch - plot_from_epoch), mean_val_acc[plot_from_epoch:plot_to_epoch], color='r', label='validation')
    plt.fill_between(range(plot_to_epoch - plot_from_epoch), np.maximum(0, mean_val_acc[plot_from_epoch:plot_to_epoch] - std_val_acc[plot_from_epoch:plot_to_epoch]),
                     np.minimum(1, mean_val_acc[plot_from_epoch:plot_to_epoch] + std_val_acc[plot_from_epoch:plot_to_epoch]), alpha=0.2, color='r')

    if multival is not None:
        for i in range(m):
            plt.plot(range(plot_to_epoch - plot_from_epoch), mean_val_acc_scales[plot_from_epoch:plot_to_epoch, i], label=f'validation_{i+1}')
            plt.fill_between(range(plot_to_epoch - plot_from_epoch), np.maximum(0, mean_val_acc_scales[plot_from_epoch:plot_to_epoch, i] - std_val_acc_scales[plot_from_epoch:plot_to_epoch, i]),
                             np.minimum(1, mean_val_acc_scales[plot_from_epoch:plot_to_epoch, i] + std_val_acc_scales[plot_from_epoch:plot_to_epoch, i]), alpha=0.2)

    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend()

    if filepath is not None:
        plt.savefig(filepath + model_n + '_acc' + '.png')

    if IoU_in_history:
        plt.figure(figsize=(10, 10))
        plt.title(model_n + '_mIoU')

        plt.ylim(0, 1.05)
        plt.plot(range(plot_to_epoch - plot_from_epoch), mean_train_mIoU[plot_from_epoch:plot_to_epoch], color="g",
                 label='training')
        plt.fill_between(np.arange(plot_to_epoch - plot_from_epoch), np.maximum(0, mean_train_mIoU[plot_from_epoch:plot_to_epoch] - std_train_mIoU[plot_from_epoch:plot_to_epoch]),
                         np.minimum(1, mean_train_mIoU[plot_from_epoch:plot_to_epoch] + std_train_mIoU[plot_from_epoch:plot_to_epoch]),
                         alpha=0.2, color="g")

        plt.plot(range(plot_to_epoch - plot_from_epoch), mean_val_mIoU[plot_from_epoch:plot_to_epoch], color='r',
                 label='validation')
        plt.fill_between(range(plot_to_epoch - plot_from_epoch), np.maximum(0, mean_val_mIoU[plot_from_epoch:plot_to_epoch] - std_val_mIoU[plot_from_epoch:plot_to_epoch]),
                         np.minimum(1, mean_val_mIoU[plot_from_epoch:plot_to_epoch] + std_val_mIoU[plot_from_epoch:plot_to_epoch]),
                         alpha=0.2, color='r')

        plt.xlabel("Epoch number")
        plt.ylabel("mIoU")
        plt.legend()

    if filepath is not None:
        if IoU_in_history:
            plt.savefig(filepath + model_n + '_mIoU' + '.png')
    else:
        plt.show()

    plt.close('all')

    return mean_train_loss, std_train_loss, mean_train_acc, std_train_acc, mean_val_loss, std_val_loss, mean_val_acc, std_val_acc