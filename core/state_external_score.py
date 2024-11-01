import os
import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn
import collections

from torchvision.transforms import transforms

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data directory')
    parser.add_argument('--theta', default='', type=float, help='theta')
    parser.add_argument('--mask_gt_dir', default='', type=str, help='ground-truth mask directory')
    parser.add_argument('--mask_pd_dir', default='', type=str, help='predicted mask directory')
    parser.add_argument('--save_dir', default='', type=str, help='save directory')
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
    print('DATA PATH:', args.data_dir)
    print('MASK PATH:', args.mask_pd_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    state = torch.load(args.model_path)
    # state = torch.load(args.model_path)['model']
    if isinstance(state, collections.OrderedDict):
        model = models.load_model(args.model_name, num_classes=args.num_classes)
        model.load_state_dict(state)
    else:
        model = state
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------
    # each epoch
    # ----------------------------------------

    value_types = ['ag']
    cache_types = ['x', 'c', 's', 'z', 'h']
    # cache_types = ['x']
    layers = [l for l in range(len(model.layers))]

    # #################### iou ####################
    # each cache_type + value_type
    scores_all_iou = []
    for value_type in value_types:
        for cache_type in cache_types:
            # each layer
            scores_iou = []
            for layer in layers:
                # mask_pd_dir=${result_path}'/'${exp_name}'/features/?seqs/external_score/{}/layer{}'
                mask_pd_dir = args.mask_pd_dir.format(cache_type + '-' + value_type, layer)
                print(args.mask_gt_dir)
                print(mask_pd_dir)
                iou = test_iou(args.mask_gt_dir, mask_pd_dir, args.theta)
                print(iou)
                scores_iou.append(iou)  # (l)

            scores_all_iou.append(scores_iou)  # (type, l)

    scores_all_iou = np.asarray(scores_all_iou)

    # ------------------ lineplot ------------------
    # fig_path = os.path.join(args.save_dir, '{}_{}_{}.png'.format(str(value_types), 'iou', str(args.theta)))
    # draw_line(scores_all_iou, fig_path)
    fig_path = os.path.join(args.save_dir, '{}_{}_{}.npy'.format(str(value_types), 'iou', str(args.theta)))
    np.save(fig_path, scores_all_iou)
    # ------------------ lineplot ------------------
    # #################### iou ####################

    #################### score ####################
    # each cache_type + value_type
    scores_all_neg = []
    scores_all_pos = []
    scores_all_cov = []
    for value_type in value_types:
        for cache_type in cache_types:
            # each layer
            scores_neg = []
            scores_pos = []
            scores_cov = []
            for layer in layers:
                # mask_pd_dir=${result_path}'/'${exp_name}'/features/hseqs/external_score/{}/layer{}'
                mask_pd_dir = args.mask_pd_dir.format(cache_type + '-' + value_type, layer)
                print(mask_pd_dir)
                test_loader = loaders.load_data_mask(args.data_dir,
                                                     mask_pd_dir,
                                                     args.data_name,
                                                     data_type='test',
                                                     batch_size=256)
                score_neg, score_pos, score_cov = test_score(test_loader, model, criterion, args.theta, device)

                scores_neg.append(score_neg)
                scores_pos.append(score_pos)
                scores_cov.append(score_cov)

            scores_all_neg.append(scores_neg)  # (types, layers)
            scores_all_pos.append(scores_pos)
            scores_all_cov.append(scores_cov)

    scores_all_neg = np.asarray(scores_all_neg)
    scores_all_pos = np.asarray(scores_all_pos)
    scores_all_cov = np.asarray(scores_all_cov)

    # ------------------ lineplot ------------------
    # fig_path = os.path.join(args.save_dir, '{}_{}_{}.png'.format(str(value_types), 'neg', str(args.theta)))
    # draw_line(scores_all_neg, fig_path)
    # fig_path = os.path.join(args.save_dir, '{}_{}_{}.png'.format(str(value_types), 'pos', str(args.theta)))
    # draw_line(scores_all_pos, fig_path)
    # fig_path = os.path.join(args.save_dir, '{}_{}_{}.png'.format(str(value_types), 'cov', str(args.theta)))
    # draw_line(scores_all_cov, fig_path)
    fig_path = os.path.join(args.save_dir, '{}_{}_{}.npy'.format(str(value_types), 'neg', str(args.theta)))
    np.save(fig_path, scores_all_neg)
    fig_path = os.path.join(args.save_dir, '{}_{}_{}.npy'.format(str(value_types), 'pos', str(args.theta)))
    np.save(fig_path, scores_all_pos)
    fig_path = os.path.join(args.save_dir, '{}_{}_{}.npy'.format(str(value_types), 'cov', str(args.theta)))
    np.save(fig_path, scores_all_cov)
    # ------------------ lineplot ------------------
    #################### score ####################

    # #################### all ####################
    # scores_all_sal = (scores_all_neg + 1e-5) / (scores_all_pos + 1e-5)
    # fig_path = os.path.join(args.save_dir, '{}_{}_{}.png'.format(str(value_types), 'sal', str(args.theta)))
    # draw_line(scores_all_sal, fig_path)
    # scores_all = scores_all_sal * scores_all_iou
    # fig_path = os.path.join(args.save_dir, '{}_{}.png'.format(str(value_types), str(args.theta)))
    # draw_line(scores_all, fig_path)
    # #################### all ####################

    # print(np.mean(scores_all_sal, axis=1), np.mean(scores_all_cov, axis=1), np.mean(scores_all, axis=1), )
    print('-' * 50)
    print('COMPLETE !!!')


def load_mask(mask_path, input_size=224):
    t = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    mask = Image.open(mask_path).convert('L')
    mask = t(mask).unsqueeze(0)  # (c, h, w)
    # print(mask.shape)
    # print(torch.mean(mask), torch.min(mask), torch.max(mask))

    return mask


def test_iou(gt_dir, pd_dir, theta):
    scores = []
    for root, dirs, files in os.walk(gt_dir):
        for f in files:
            gt_path = os.path.join(gt_dir, f)
            gt_mask = load_mask(gt_path).bool()

            pd_path = os.path.join(pd_dir, f)
            pd_mask = load_mask(pd_path)
            pd_mask = torch.where(pd_mask > float(theta), 1, 0).bool()  # (b, 1, h, w)

            # Calculate intersection
            intersection = torch.logical_and(gt_mask, pd_mask).float().sum()
            union = torch.logical_or(gt_mask, pd_mask).float().sum()
            iou = (intersection + 1e-5) / (union + 1e-5)

            scores.append(iou)
    scores = torch.mean(torch.asarray(scores)).item()
    # print(scores)
    return scores


def draw_line(scores, fig_path):
    plt.figure(figsize=(9, 6))
    df = pd.DataFrame(scores).transpose()
    sns.set(style="whitegrid")
    sns.lineplot(data=df)
    # plt.legend(labels=cache_types)
    plt.savefig(fig_path, bbox_inches='tight')
    # plt.legend(cache_types)
    plt.clf()
    plt.close()


def test_score(test_loader, model, criterion, theta, device):
    # acc1_meter = AverageMeter('Acc@1', ':6.2f')
    # progress = ProgressMeter(total=len(test_loader), step=20, prefix='Test',
    #                          meters=[acc1_meter])
    model.eval()

    scores_neg = []
    scores_pos = []
    scores_cov = []
    for i, samples in enumerate(test_loader):
        inputs, labels, masks = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(False):
            # print(masks.shape, torch.min(masks), torch.max(masks))

            masks = torch.where(masks > theta, 1.0, 0.0)  # (b, 1, h, w)
            # --------------------------------------------
            score_c = torch.mean(masks)  # 1
            scores_cov.append(score_c)
            # --------------------------------------------
            inputs_n = inputs * masks
            outputs_n = model(inputs_n)  # (b, c)
            scores_n = torch.softmax(outputs_n, dim=1)  # (b, c)
            scores_n = torch.gather(scores_n, index=labels.unsqueeze(1), dim=1)  # (b)
            # scores_n = scores_n / (torch.mean(masks, dim=(1, 2, 3)) + 1e-5)
            scores_n = torch.mean(scores_n)  # 1
            scores_neg.append(scores_n)
            # --------------------------------------------
            inputs_p = inputs * (1 - masks)
            outputs_p = model(inputs_p)  # (b, c)
            scores_p = torch.softmax(outputs_p, dim=1)  # (b, c)
            scores_p = torch.gather(scores_p, index=labels.unsqueeze(1), dim=1)  # (b)
            # scores_p = scores_p / (torch.mean((1 - masks), dim=(1, 2, 3)) + 1e-5)
            scores_p = torch.mean(scores_p)  # 1
            scores_pos.append(scores_p)
            # --------------------------------------------
            # score_c = torch.mean(x1 / (x2 + 1e-5))
            # scores_cov.append(score_c)

            # # loss = criterion(outputs, labels)
            # acc1, = metrics.accuracy(outputs, labels)
            # acc1_meter.update(acc1.item(), inputs.size(0))
            # progress.display(i)

    scores_neg = torch.mean(torch.asarray(scores_neg)).item()
    scores_pos = torch.mean(torch.asarray(scores_pos)).item()
    scores_cov = torch.mean(torch.asarray(scores_cov)).item()

    return scores_neg, scores_pos, scores_cov


if __name__ == '__main__':
    main()
