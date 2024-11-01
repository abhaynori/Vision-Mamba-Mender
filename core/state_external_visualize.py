import argparse
import os
import pickle

import torch
import numpy as np
from PIL import Image

from core.mamba_mech import spatial_heatmap, save_spatial_heatmaps


def mm_norm(a, dim=-1):
    a_min = torch.min(a, dim=dim, keepdim=True)[0]
    a_max = torch.max(a, dim=dim, keepdim=True)[0]
    a_normalized = (a - a_min) / (a_max - a_min + 1e-5)

    return a_normalized


def load_data(data_path):
    data_file = open(data_path, 'rb')
    data_layer = pickle.load(data_file)  # (c, n, d, s)
    data_layer = torch.from_numpy(data_layer)
    return data_layer


def load_names(sample_dir, sample_name_path=None):
    if sample_name_path is not None:
        name_file = open(sample_name_path, 'rb')
        sample_names = pickle.load(name_file)  # (c, n)
        class_names = sorted([d.name for d in os.scandir(sample_dir) if d.is_dir()])  # (c)
        return sample_names, class_names
    else:
        class_names = sorted([d.name for d in os.scandir(sample_dir) if d.is_dir()])  # (c)
        return class_names


def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img = torch.from_numpy(np.array(img))
    # print('img.shape:', img.shape)
    return img


def load_state_external(data_dir, layer, cache_type):  # a, g, ag(recommend)
    if '0' in cache_type:
        data_a = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '-a')))
        data_g = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '-g')))
    elif '1' in cache_type:
        data_a = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '-a'))).flip(-1)
        data_g = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '-g'))).flip(-1)
    elif 'h' == cache_type:
        data_a = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(
            layer, cache_type + '-a'))).permute(0, 1, 3, 2)
        data_g = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(
            layer, cache_type + '-g'))).permute(0, 1, 3, 2)
    else:
        data_a0 = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '0-a')))
        data_g0 = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '0-g')))
        data_a1 = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '1-a'))).flip(-1)
        data_g1 = load_data(os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '1-g'))).flip(-1)
        data_a = data_a0 + data_a1
        data_g = data_g0 + data_g1

    data_a = torch.relu(data_a)  # (c, n, d, s)
    data_g = torch.relu(data_g)  # (c, n, d, s)
    data_g = torch.mean(data_g, dim=3, keepdim=True)  # (c, n, d, 1)

    data = data_a * data_g  # (c, n, d, s)

    return data


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_layers', default='', type=int, help='num layers')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--num_samples', default='', type=int, help='num samples')
    parser.add_argument('--cache_types', default=None, nargs='+', type=str, help='cache types')
    parser.add_argument('--sample_dir', default='', type=str, help='sample dir')
    parser.add_argument('--sample_name_path', default='', type=str, help='sample name path')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    layers = [l for l in range(args.num_layers)]  # TODO Not necessarily all layers
    # labels = [c for c in range(args.num_classes)]  # TODO Not necessarily all classes
    labels = [9, 2, 0]
    cls_pos = 98  # TODO Are there classification tokens?

    sample_names, class_names = load_names(args.sample_dir, args.sample_name_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cams = [[[] for _ in range(args.num_samples)] for _ in range(len(labels))]
    for cache_type in args.cache_types:
        for layer in layers:
            #################### data ####################
            print('===> load data for layer', layer)
            data = load_state_external(args.data_dir, layer, cache_type)
            #################### data ####################

            for label_idx, label in enumerate(labels):
                data_c = data[label]  # (n, d, s)
                data_c = torch.mean(data_c, dim=1)  # (n, s)

                #################### cam ####################
                print('--> generate cam for label', label)
                data_c = torch.cat((data_c[:, :cls_pos], data_c[:, (cls_pos + 1):]), dim=-1)  # (n, s-1)
                for sample_idx in range(args.num_samples):
                    img_path = os.path.join(args.sample_dir, class_names[label], sample_names[label][sample_idx])
                    img = load_image(img_path)
                    mask = data_c[sample_idx]
                    cam = spatial_heatmap(img, mask)
                    cams[label_idx][sample_idx].append(cam)

        for label_idx, label in enumerate(labels):
            for sample_idx in range(args.num_samples):
                img_path = os.path.join(args.sample_dir, class_names[label], sample_names[label][sample_idx])
                img = load_image(img_path)

                save_spatial_heatmaps(figs=[img] + cams[label_idx][sample_idx],
                                      save_dir=args.save_dir, class_name=class_names[label],
                                      img_name=sample_names[label][sample_idx], suffix=cache_type + '-ag')
                #################### cam ####################


if __name__ == '__main__':
    main()
