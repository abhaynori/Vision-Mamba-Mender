import os
import argparse
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import collections

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter

from PIL import Image
import torchvision.transforms as transforms
from core.xai_utils import *


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--image_path', default='', type=str, help='image path')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('TEST ON:', device)
    print('MODEL PATH:', args.model_path)
    print('SAVE DIR:', args.save_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    # model configuration
    state = torch.load(args.model_path)['model']
    # state = torch.load(args.model_path)
    if isinstance(state, collections.OrderedDict):
        model = models.load_model(args.model_name, num_classes=args.num_classes)
        model.load_state_dict(state)
    else:
        model = state
    model.to(device)
    model.eval()

    # data configuration
    image = load_single_image(args.image_path)
    image = image.to(device)
    print('-->', args.image_path)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    map_raw_atten, logits = generate_raw_attn(model, image, start_layer=20)  # (1, num_patches)
    map_rollout, _ = generate_rollout(model, image, start_layer=20)  # (1, num_patches)
    map_mamba_attr, _ = generate_mamba_attr(model, image, start_layer=20)  # (1, num_patches)

    raw_image = Image.open(args.image_path)
    raw_attn = generate_visualization(inverse_image(image), map_raw_atten)
    rollout = generate_visualization(inverse_image(image), map_rollout)
    mamba_attr = generate_visualization(inverse_image(image), map_mamba_attr)
    # print logits
    fig, axs = plt.subplots(1, 4, figsize=(10, 10))
    axs[0].imshow(raw_image)
    axs[0].axis('off')
    axs[1].imshow(raw_attn)
    axs[1].axis('off')
    axs[2].imshow(rollout)
    axs[2].axis('off')
    axs[3].imshow(mamba_attr)
    axs[3].axis('off')

    fig_path = os.path.join(args.save_dir, os.path.split(args.image_path)[-1])
    print('-->', fig_path)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()

    print('-' * 50)
    print('COMPLETE !!!')


def load_single_image(image_path, input_size=224):
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    transform_eval = transforms.Compose([
        transforms.Resize(int(input_size)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    img = Image.open(image_path).convert('RGB')
    transformed_img = transform_eval(img).unsqueeze(0)  # (c, h, w)
    # transformed_img = transformed_img.permute(1, 2, 0)  # (h, w, c)

    return transformed_img


def inverse_image(image):
    image = image.squeeze()
    image = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1., 1., 1.]),
    ])(image)
    image = image.detach().cpu()
    return image


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(original_image, transformer_attribution):
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


if __name__ == '__main__':
    main()
