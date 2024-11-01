import os
import shutil

import cv2
import torchvision

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torchvision.transforms as transforms
import torch


def dilate_mask():
    input_dir = '/nfs196/hjc/datasets/ImageNet-Seg/'
    output_dir = '/nfs196/hjc/datasets/ImageNet-Seg@10'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            input_path = os.path.join(root, f)
            output_path = os.path.join(output_dir, f)
            print(input_path, '->', output_path)

            # 值变换
            img_ori = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # # 前景标注值转255
            # # img_mean = np.mean(img)
            # img = np.where(img >= 1, 255, 0)
            # cv2.imwrite(output_path, img.astype(np.uint8))

            # 膨胀 + 前景标注值转255
            kernel = np.ones((3, 3), np.uint8)
            img_tgt = cv2.dilate(img_ori, kernel, iterations=10)
            # img_tgt = np.where(img_tgt >= 1, 255, 0)
            # print(img_ori.shape, img_ori.min(), img_ori.max(), '-->', img_tgt.shape, img_tgt.min(), img_tgt.max())
            cv2.imwrite(output_path, img_tgt.astype(np.uint8))


def select_image_match_mask():
    # mask_dir = '/nfs/ch/project/td/output/ideal/atts/imagenet50'
    # src_dir = '/nfs/ch/project/td/dataset/imagenet50/train'
    # dst_dir = '/nfs/ch/project/td/dataset/imagenet50/train-m'
    mask_dir = '/nfs/ch/project/td/output/ideal/atts/imagenet10'
    src_dir = '/nfs/ch/project/td/dataset/imagenet10/train'
    dst_dir = '/nfs/ch/project/td/dataset/imagenet10/train-m'
    for root, dirs, files in os.walk(mask_dir):
        for f in files:
            fname, ext = os.path.splitext(f)
            class_name = f.split('_')[0]
            src_path = os.path.join(src_dir, class_name, fname + '.JPEG')
            dst_path = os.path.join(dst_dir, class_name, fname + '.JPEG')
            print(src_path, dst_path)
            copy_file(src_path, dst_path)


def check_input_mask(inputs, masks, fig_dir, batch):
    # print(inputs.shape, "|", masks.shape, torch.min(masks), torch.max(masks))

    imgs = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1., 1., 1.]),
    ])(inputs)
    imgs = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
    masks = masks.permute(0, 2, 3, 1).detach().cpu().numpy()[:, :, :, 0] * 255
    # print('masks', masks.shape, np.min(masks), np.max(masks))

    fig, axs = plt.subplots(10, 3, figsize=(3, 10))
    n = 0
    g = 0  # ten a group
    for i in range(len(masks)):
        if np.min(masks[i]) < 100 and n < 10:
            # if n < 10:
            print('-->', n, ':', i, "|", np.min(masks[i]), np.max(masks[i]))
            axs[n][0].imshow(imgs[i])
            axs[n][0].axis('off')
            axs[n][1].imshow(masks[i], cmap=plt.cm.gray, vmin=0, vmax=255)
            axs[n][1].axis('off')
            bn_mask = np.stack([masks[i]] * 3, axis=-1) // 255
            axs[n][2].imshow(imgs[i] * bn_mask)
            axs[n][2].axis('off')
            n += 1

        if n == 10:
            fig_path = os.path.join(fig_dir, 'batch{}_group{}.png'.format(batch, g))
            print(fig_path)
            plt.savefig(fig_path, bbox_inches='tight')
            plt.clf()
            plt.close()
            fig, axs = plt.subplots(10, 2, figsize=(2, 10))
            n = 0
            g += 1

    if 0 < n < 10:
        fig_path = os.path.join(fig_dir, 'batch{}_group{}.png'.format(batch, g))
        print(fig_path)
        plt.savefig(fig_path, bbox_inches='tight')
        plt.clf()
        plt.close()


def copy_file(src, dst):
    path, name = os.path.split(dst)
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.copyfile(src, dst)


if __name__ == '__main__':
    dilate_mask()
    # select_image_match_mask()
