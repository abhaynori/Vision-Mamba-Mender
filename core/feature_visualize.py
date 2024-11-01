import os
import pickle

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from PIL import Image

from core.mamba_mech import spatial_heatmap, save_spatial_heatmaps


# from core.mamba_cam import load_single_image, inverse_image

def mm_norm(a, dim=-1):
    a_min = torch.min(a, dim=dim, keepdim=True)[0]
    a_max = torch.max(a, dim=dim, keepdim=True)[0]
    a_normalized = (a - a_min) / (a_max - a_min + 1e-5)

    return a_normalized

def _load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img = torch.from_numpy(np.array(img))
    # print('img.shape:', img.shape)
    return img


def _load_names(exp_name, input_dir, name_path):
    name_file = open(name_path.format(exp_name), 'rb')
    sample_names = pickle.load(name_file)  # (l, c, n)
    class_names = sorted([d.name for d in os.scandir(input_dir) if d.is_dir()])  # (c)

    return sample_names, class_names


def _load_data(exp_name, data_name, layer, data_path):
    data_file = open(data_path.format(exp_name, layer, data_name), 'rb')
    data_layer = pickle.load(data_file)  # (c, n, d, s)
    data_layer = torch.from_numpy(data_layer)
    return data_layer


def _load_datas(exp_name, cache_type, value_type, layer, data_path, if_weight_all=True):
    if '0' in cache_type:
        data_a = _load_data(exp_name, cache_type + '-a', layer, data_path)  # (c, n, d, s)
        data_g = _load_data(exp_name, cache_type + '-g', layer, data_path)  # (c, n, d, s)
    elif '1' in cache_type:
        data_a = _load_data(exp_name, cache_type + '-a', layer, data_path).flip(-1)  # (c, n, d, s)
        data_g = _load_data(exp_name, cache_type + '-g', layer, data_path).flip(-1)  # (c, n, d, s)
    elif 'h' == cache_type:
        data_a = _load_data(exp_name, cache_type + '-a', layer, data_path).permute(0, 1, 3, 2)  # (c, n, d, s)
        data_g = _load_data(exp_name, cache_type + '-g', layer, data_path).permute(0, 1, 3, 2)  # (c, n, d, s)
    else:
        data_a0 = _load_data(exp_name, cache_type + '0-a', layer, data_path)  # (c, n, d, s)
        data_g0 = _load_data(exp_name, cache_type + '0-g', layer, data_path)  # (c, n, d, s)
        data_a1 = _load_data(exp_name, cache_type + '1-a', layer, data_path).flip(-1)  # (c, n, d, s)
        data_g1 = _load_data(exp_name, cache_type + '1-g', layer, data_path).flip(-1)  # (c, n, d, s)
        data_a = data_a0 + data_a1
        data_g = data_g0 + data_g1

    data_a = torch.relu(data_a)
    data_g = torch.relu(data_g)

    data = None
    if 'a' == value_type:
        data = data_a
    elif 'g' == value_type:
        data = data_g
    elif 'ag' == value_type:
        if not if_weight_all:
            data_g = torch.mean(data_g, dim=3, keepdim=True)  # (c, n, d, 1)
        data = data_a * data_g  # (c, n, d, s)
    # print('data size', data.shape)

    return data


def _external_vis(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
                  input_dir, data_path, name_path, save_dir):
    img_dir = os.path.join(save_dir.format(exp_name), 'external')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    save_suffix = cache_type + '-' + value_type
    names, class_names = _load_names(exp_name, input_dir, name_path)
    cams = [[[] for _ in range(num_samples)] for _ in range(len(labels))]

    for layer in layers:
        #################### data ####################
        print('===> load data for layer', layer)
        data = _load_datas(exp_name, cache_type, value_type, layer, data_path, if_weight_all=False)
        #################### data ####################

        for label_idx, label in enumerate(labels):
            data_c = data[label]  # (n, d, s)
            data_c = torch.mean(data_c, dim=1)  # (n, s)

            # #################### heatmap ####################
            # print('--> heat')
            # fig_path = os.path.join(save_dir.format(exp_name),
            #                         'layer{}_label{}_{}.png'.format(layer, label, data_name))
            # plt.figure(figsize=(100, 20))
            # sns.heatmap(data=data.numpy(), annot=False)
            # plt.savefig(fig_path, bbox_inches='tight')
            # plt.clf()
            # plt.close()
            # #################### heatmap ####################

            #################### cam ####################
            print('--> generate cam for label', label)
            data_c = torch.cat((data_c[:, :cls_pos], data_c[:, (cls_pos + 1):]), dim=-1)  # (n, s-1)
            for sample_idx in range(len(data_c)):
                img_path = os.path.join(input_dir, class_names[label], names[label][sample_idx])
                img = _load_image(img_path)
                mask = data_c[sample_idx]
                cam = spatial_heatmap(img, mask)
                cams[label_idx][sample_idx].append(cam)

    for label_idx, label in enumerate(labels):
        for sample_idx in range(num_samples):
            img_path = os.path.join(input_dir, class_names[label], names[label][sample_idx])
            img = _load_image(img_path)

            save_spatial_heatmaps(figs=[img] + cams[label_idx][sample_idx],
                                  save_dir=img_dir, class_name=class_names[label],
                                  img_name=names[label][sample_idx], suffix=save_suffix)

            #################### cam ####################


def _external_mask(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
                   input_dir, data_path, name_path, save_dir):
    names, class_names = _load_names(exp_name, input_dir, name_path)

    for layer in layers:

        img_dir = os.path.join(save_dir.format(exp_name), 'external_score',
                               cache_type + '-' + value_type, 'layer{}'.format(layer))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        #################### data ####################
        print('===> load data for layer', layer)
        data = _load_datas(exp_name, cache_type, value_type, layer, data_path, if_weight_all=False)
        #################### data ####################

        for label_idx, label in enumerate(labels):
            data_c = data[label]  # (n, d, s)
            data_c = torch.mean(data_c, dim=1)  # (n, s)

            #################### cam ####################
            print('--> generate mask for label', label)
            data_c = torch.cat((data_c[:, :cls_pos], data_c[:, (cls_pos + 1):]), dim=-1)  # (n, s-1)
            for sample_idx in range(len(data_c)):
                mask = data_c[sample_idx]  # (s-1)
                mask = mask.reshape(1, 1, 14, 14)
                mask = torch.nn.functional.interpolate(mask, scale_factor=16, mode='bilinear')
                mask = mask.reshape(224, 224)
                mask = (mask - mask.min()) / (mask.max() - mask.min())
                mask = mask.numpy()
                mask = np.uint8(255 * mask)
                # mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                gray = Image.fromarray(mask)
                gray = gray.convert('L')
                name, _ = os.path.splitext(names[label][sample_idx])
                img_path = os.path.join(img_dir, name + '.png')
                gray.save(img_path)
            #################### cam ####################


def _internal_vis(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
                  input_dir, data_path, name_path, save_dir):
    names, class_names = _load_names(exp_name, input_dir, name_path)

    for layer in layers:
        #################### data ####################
        print('===> load data for layer', layer)
        data = _load_datas(exp_name, cache_type, value_type, layer, data_path, if_weight_all=True)
        #################### data ####################

        for label_idx, label in enumerate(labels):
            print('--> label', label)
            data_c = data[label]  # (n, d, s)
            # data_c = torch.mean(data_c, dim=2)  # (n, d)
            data_c = data_c[:, :, cls_pos]  # (n, d)
            data_c = data_c.numpy()  # (n, d)

            #################### heatmap ####################
            fig_dir = os.path.join(save_dir.format(exp_name), 'internal', class_names[label])
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            fig_path = os.path.join(fig_dir, 'layer{}_{}.png'.format(layer, cache_type + '-' + value_type))
            plt.figure(figsize=(100, 10))
            sns.heatmap(data=data_c, annot=False)
            plt.savefig(fig_path, bbox_inches='tight')
            plt.clf()
            plt.close()
            #################### heatmap ###################


def _inernal_score(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
                   input_dir, data_path, name_path, save_dir):
    scoresD = [[] for _ in range(len(labels))]
    scoresS = [[] for _ in range(len(labels))]
    scores = [[] for _ in range(len(labels))]
    for layer in layers:
        #################### data ####################
        print('===> load data for layer', layer)
        data = _load_datas(exp_name, cache_type, value_type, layer, data_path, if_weight_all=True)
        #################### data ####################

        for label_idx, label in enumerate(labels):
            print('--> label', label)
            data_c = data[label]  # (n, d, s)
            # data_c = torch.mean(data_c, dim=2)  # (n, d)
            data_c = data_c[:, :, cls_pos]  # (n, d)
            # ---------------------------------
            data_c = mm_norm(data_c, dim=-1)  # (n, d)
            alpha = 0.3
            data_c = torch.where(data_c > alpha, 1, 0).to(torch.float)  # (n, d)
            # ---------------------------------

            # #################### heatmap ####################
            # fig_dir = os.path.join(save_dir.format(exp_name), 'internal_score', class_names[label])
            # if not os.path.exists(fig_dir):
            #     os.makedirs(fig_dir)
            # fig_path = os.path.join(fig_dir, 'layer{}_{}.png'.format(layer, save_suffix))
            # plt.figure(figsize=(100, 10))
            # sns.heatmap(data=data_c, annot=False)
            # plt.savefig(fig_path, bbox_inches='tight')
            # plt.clf()
            # plt.close()
            # #################### heatmap ###################

            #################### score ####################
            den = torch.mean(data_c)  # (n, d) -> ()
            sim_d = torch.where(torch.sum(data_c, dim=0, keepdim=True) > 0, 1, 0)  # (1, d)
            sim = torch.sum(torch.mean(data_c, dim=0)) / torch.sum(sim_d)
            score = sim / (den + 1e-5)
            scoresD[label_idx].append(round(den.item(), 2))
            scoresS[label_idx].append(round(sim.item(), 2))
            scores[label_idx].append(round(score.item(), 2))
            #################### score ####################
    # print(scoresD)
    # print('-' * 10)
    # print(scoresS)
    # print('-' * 10)
    # print(scores)
    return scores


def _inernal_mask(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
                  input_dir, data_path, name_path, save_dir):
    for layer in layers:
        #################### data ####################
        print('===> load data for layer', layer)
        data = _load_datas(exp_name, cache_type, value_type, layer, data_path, if_weight_all=True)
        #################### data ####################

        masks = []
        for label_idx, label in enumerate(labels):
            print('--> label', label)
            data_c = data[label]  # (n, d, s)
            # data_c = torch.mean(data_c, dim=2)  # (n, d)
            data_c = data_c[:, :, cls_pos]  # (n, d)
            # ---------------------------------
            mask_c = torch.mean(data_c, dim=0, keepdim=True)  # (n, d) -> (1, d)
            mask_c = mm_norm(mask_c, dim=-1)  # (1, d)
            alpha = 0.3
            mask_c = torch.where(mask_c > alpha, 1, 0)  # (1, d)
            masks.append(mask_c)
            # ---------------------------------
        masks = torch.cat(masks, dim=0)  # (c, d)
        mask_dir = os.path.join(save_dir.format(exp_name), 'internal_score', 'masks')
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
        mask_path = os.path.join(mask_dir, 'layer{}_{}.pt'.format(layer, cache_type + '-' + value_type))
        print('->', mask_path)
        torch.save(masks, mask_path)

        # #################### heatmap ####################
        # masks = masks.numpy()
        # fig_dir = os.path.join(save_dir.format(exp_name), 'internal_score', 'figs')
        # if not os.path.exists(fig_dir):
        #     os.mkdir(fig_dir)
        # mask_fig_path = os.path.join(fig_dir, 'layer{}_{}.png'.format(layer, cache_type + '-' + value_type))
        # plt.figure(figsize=(100, 10))
        # sns.heatmap(data=masks, annot=False)
        # plt.savefig(mask_fig_path, bbox_inches='tight')
        # plt.clf()
        # plt.close()
        # #################### heatmap ###################


def vis_seqs():
    # data
    exp_name = 'vimd12_imagenet10'

    # image name
    # input_dir = '/nfs3/hjc/projects/MD/outputs/vimd12_imagenet10/images/htrain'
    # data_path = '/nfs3/hjc/projects/MD/outputs/{}/features/hdata/layer{}_{}.pkl'
    # name_path = '/nfs3/hjc/projects/MD/outputs/{}/features/hdata/sample_names.pkl'
    # save_dir = '/nfs3/hjc/projects/MD/outputs/{}/features/hseqs'
    # ----------------------------------------------------------------------------
    # input_dir = '/nfs3/hjc/projects/MD/outputs/vimd12_imagenet10/images/ltrain'
    # data_path = '/nfs3/hjc/projects/MD/outputs/{}/features/ldata/layer{}_{}.pkl'
    # name_path = '/nfs3/hjc/projects/MD/outputs/{}/features/ldata/sample_names.pkl'
    # save_dir = '/nfs3/hjc/projects/MD/outputs/{}/features/lseqs'
    # ----------------------------------------------------------------------------
    input_dir = '/nfs/ch/project/td/dataset/imagenet10/train-m'
    data_path = '/nfs3/hjc/projects/MD/outputs/{}/features/mdata/layer{}_{}.pkl'
    name_path = '/nfs3/hjc/projects/MD/outputs/{}/features/mdata/sample_names.pkl'
    save_dir = '/nfs3/hjc/projects/MD/outputs/{}/features/mseqs'

    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # labels = [9, 3, 0]
    labels = [l for l in range(10)]
    num_samples = 10
    cls_pos = 98

    # #################### state external ####################
    # # cache_types = ['x0', 'c0', 's0', 'z0', 'y0', 'x1', 'c1', 's1', 'z1', 'y1', 'x', 'c', 's', 'z', 'y', 'h']
    # # value_types = ['a', 'g', 'ag']
    # cache_types = ['x', 'c', 's', 'z', 'y', 'h']
    # value_types = ['ag']
    # for cache_type in cache_types:
    #     for value_type in value_types:
    #         _external_vis(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
    #                       input_dir, data_path, name_path, save_dir)
    # #################### state external ####################

    #################### state external mask ####################
    # cache_types = ['x0', 'c0', 's0', 'z0', 'y0', 'x1', 'c1', 's1', 'z1', 'y1', 'x', 'c', 's', 'z', 'y', 'h']
    # value_types = ['a', 'g', 'ag']
    cache_types = ['x', 'c', 's', 'z', 'h']
    value_types = ['ag']
    for cache_type in cache_types:
        for value_type in value_types:
            _external_mask(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
                           input_dir, data_path, name_path, save_dir)
    #################### state external mask ####################

    # #################### state internal ####################
    # cache_types = ['x0', 'c0', 's0', 'z0', 'y0', 'x1', 'c1', 's1', 'z1', 'y1', 'x', 'c', 's', 'z', 'y', 'h']
    # value_types = ['ag']
    # # cache_types = ['x0', 'c0', 's0', 'z0', 'y0', 'h']
    # # value_types = ['ag']
    # for value_type in value_types:
    #     for cache_type in cache_types:
    #         _internal_vis(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
    #                       input_dir, data_path, name_path, save_dir)
    # #################### state internal ####################

    # #################### state internal score ####################
    # scores_all = []  # (group, x_num)
    #
    # cache_types = ['x0', 'c0', 's0', 'z0', 'y0', 'h']
    # value_types = ['ag']
    # for value_type in value_types:
    #     for cache_type in cache_types:
    #         scores = _inernal_score(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
    #                                 input_dir, data_path, name_path, save_dir)
    #         scores_all.append(scores[0])
    #
    # # ------------------------------------------------------------
    # scores_all = np.asarray(scores_all)
    # fig_dir = os.path.join(save_dir.format(exp_name), 'internal_score')
    # if not os.path.exists(fig_dir):
    #     os.makedirs(fig_dir)
    # fig_path = os.path.join(fig_dir, 'score.png')
    # plt.figure(figsize=(9, 6))
    # # for i, scores in enumerate(all):
    # #     sns.lineplot(x=range(len(scores)), y=scores, legend=cache_types[i])
    # df = pd.DataFrame(scores_all).transpose()
    # sns.set(style="whitegrid")
    # sns.lineplot(data=df)
    # # plt.legend(labels=cache_types)
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.legend(cache_types)
    # plt.clf()
    # plt.close()
    # #################### state internal score ####################

    # #################### state internal mask ####################
    # # cache_types = ['x0', 'c0', 's0', 'z0', 'y0', 'x1', 'c1', 's1', 'z1', 'y1', 'h']
    # # value_types = ['ag', 'g']
    # cache_types = ['x1', 'c1', 's1', 'z1', 'y1', 'h']
    # value_types = ['g']
    # for value_type in value_types:
    #     for cache_type in cache_types:
    #         _inernal_mask(exp_name, cache_type, value_type, num_samples, cls_pos, labels, layers,
    #                       input_dir, data_path, name_path, save_dir)
    # #################### state internal mask ####################


if __name__ == '__main__':
    vis_seqs()
