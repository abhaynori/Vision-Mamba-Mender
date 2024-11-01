import argparse
import os

import torch
import seaborn as sns
from matplotlib import pyplot as plt

from state_external_visualize import load_data, load_names


def load_state_internal(data_dir, layer, cache_type):  # ag
    data_a_path = os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '-a'))
    data_a = load_data(data_a_path)  # (c, n, d, s)
    data_a = torch.relu(data_a)

    data_g_path = os.path.join(data_dir, 'layer{}_{}.pkl'.format(layer, cache_type + '-g'))
    data_g = load_data(data_g_path)  # (c, n, d, s)
    data_g = torch.relu(data_g)

    if cache_type == 'h':  # TODO permute
        data_a = data_a.permute(0, 1, 3, 2)  # (c, n, s, d) -> (c, n, d, s)
        data_g = data_g.permute(0, 1, 3, 2)  # (c, n, s, d) -> (c, n, d, s)

    data = data_a * data_g  # (c, n, d, s)

    return data


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--cache_layers', default=[], nargs='*', type=int, help='cache layers')
    parser.add_argument('--cache_types', default=None, nargs='+', type=str, help='cache types')
    parser.add_argument('--sample_dir', default='', type=str, help='sample dir')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    # labels = [c for c in range(args.num_classes)]
    labels = [0]
    cls_pos = 98  # TODO Are there classification tokens?

    class_names = load_names(args.sample_dir)

    for cache_type in args.cache_types:
        for layer in args.cache_layers:
            #################### data ####################
            print('===> load data for layer', layer)
            data = load_state_internal(args.data_dir, layer, cache_type)
            #################### data ####################

            for label_idx, label in enumerate(labels):
                print('--> label', label)
                data_c = data[label]  # (n, d, s)
                # data_c = torch.mean(data_c, dim=2)  # (n, d)
                data_c = data_c[:, :, cls_pos]  # (n, d)
                data_c = data_c.numpy()  # (n, d)

                #################### heatmap ####################
                fig_dir = os.path.join(args.save_dir, class_names[label])
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)

                fig_path = os.path.join(fig_dir, 'layer{}_{}.png'.format(layer, cache_type + '-ag'))
                plt.figure(figsize=(100, 20))
                sns.heatmap(data=data_c, annot=False)
                plt.savefig(fig_path, bbox_inches='tight')
                plt.clf()
                plt.close()
                #################### heatmap ###################


if __name__ == '__main__':
    main()
