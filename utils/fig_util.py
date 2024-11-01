import os

import cv2
import torchvision

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torchvision.transforms as transforms
import torch

matplotlib.use('AGG')


# def visualize_mask():
#     import sys
#     sys.path.append(".")
#     import loaders
#     train_loader = loaders.load_data_mask(data_dir='/nfs196/hjc/datasets/ImageNet-1K/ImageNetS50/train-semi/',
#                                           mask_dir='/nfs196/hjc/datasets/ImageNet-1K/ImageNetS50/train-semi-segmentation/',
#                                           data_name='imagenet',
#                                           data_type='train',
#                                           batch_size=128,
#                                           args=None)
#     for i, samples in enumerate(train_loader):
#         inputs, labels, masks = samples
#         check_input_mask(inputs, masks, '/nfs196/hjc/projects/Mamba/outputs/test_utils/')


def heatmap(vals, fig_path, fig_w=None, fig_h=None, annot=False):
    if fig_w is None:
        fig_w = vals.shape[1]
    if fig_h is None:
        fig_h = vals.shape[0]

    f, ax = plt.subplots(figsize=(fig_w, fig_h), ncols=1)
    sns.heatmap(vals, ax=ax, annot=annot)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def imshow(img, title, fig_path):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True, nrow=10)
    npimg = img.numpy()
    # fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.title(title)
    # plt.show()

    plt.title(title)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def save_img_by_cv2(img, path):
    img_dir, _ = os.path.split(path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)


if __name__ == '__main__':
    import PIL.Image as Image
    import cv2
