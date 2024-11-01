import argparse
import os

import torch
import seaborn as sns
from matplotlib import pyplot as plt

from state_external_visualize import mm_norm
from state_internal_visualize import load_state_internal


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--cache_layers', default=[], nargs='*', type=int, help='cache layers')
    parser.add_argument('--cache_types', default=None, nargs='+', type=str, help='cache types')
    parser.add_argument('--theta', default=0.3, type=float, help='threshold')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    labels = [c for c in range(args.num_classes)]

    if 'vim' in args.model_name:  # vim, local_vim
        cls_pos = 98
    elif 'vmamba' in args.model_name:  # vmamba, efficient-vmamba
        cls_pos = None
    elif 'simba' in args.model_name:  # simba
        cls_pos = 0

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for cache_type in args.cache_types:
        for layer in args.cache_layers:
            #################### data ####################
            data = load_state_internal(args.data_dir, layer, cache_type)
            print('===> load data for layer', layer, data.shape)
            #################### data ####################

            masks = []
            for label_idx, label in enumerate(labels):
                print('--> label', label)
                data_c = data[label]  # (n, d, s)
                # data_c = torch.mean(data_c, dim=2)  # (n, d)
                if cls_pos is not None:
                    data_c = data_c[:, :, cls_pos]  # (n, d)
                else:
                    data_c = torch.mean(data_c, dim=2)  # (n,d,s)->(n,d)
                # ---------------------------------
                mask_c = torch.mean(data_c, dim=0, keepdim=True)  # (n, d) -> (1, d)
                mask_c = mm_norm(mask_c, dim=-1)  # (1, d)
                mask_c = torch.where(mask_c > args.theta, 1, 0)  # (1, d)
                masks.append(mask_c)
                # ---------------------------------
            masks = torch.cat(masks, dim=0)  # (c, d)

            mask_path = os.path.join(args.save_dir, 'layer{}_{}.pt'.format(layer, cache_type + '-ag'))
            print('->', mask_path)
            torch.save(masks, mask_path)

            #################### heatmap ####################
            masks = masks.numpy()
            fig_dir = os.path.join(args.save_dir, 'figs')
            if not os.path.exists(fig_dir):
                os.mkdir(fig_dir)

            mask_fig_path = os.path.join(fig_dir, 'layer{}_{}.png'.format(layer, cache_type + '-ag'))
            plt.figure(figsize=(100, 20))
            sns.heatmap(data=masks, annot=False)
            plt.savefig(mask_fig_path, bbox_inches='tight')
            plt.clf()
            plt.close()
            #################### heatmap ###################


if __name__ == '__main__':
    main()
