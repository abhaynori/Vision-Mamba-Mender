import argparse
import os

import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt

from state_external_visualize import mm_norm
from state_internal_visualize import load_state_internal


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_layers', default='', type=int, help='num layers')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--cache_types', default=None, nargs='+', type=str, help='cache types')
    parser.add_argument('--theta', default=0.3, type=float, help='threshold')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    layers = [l for l in range(args.num_layers)]  # TODO Not necessarily all layers
    labels = [c for c in range(args.num_classes)]
    cls_pos = 98  # TODO Are there classification tokens?

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    scores_all = []  # (group, x_num)

    for cache_type in args.cache_types:
        scoresD = [[] for _ in range(len(labels))]
        scoresS = [[] for _ in range(len(labels))]
        scores = [[] for _ in range(len(labels))]
        for layer in layers:
            #################### data ####################
            print('===> load data for layer', layer)
            data = load_state_internal(args.data_dir, layer, cache_type)
            #################### data ####################

            for label_idx, label in enumerate(labels):
                print('--> label', label)
                data_c = data[label]  # (n, d, s)
                # data_c = torch.mean(data_c, dim=2)  # (n, d)
                data_c = data_c[:, :, cls_pos]  # (n, d)
                # ---------------------------------
                data_c = mm_norm(data_c, dim=-1)  # (n, d)
                data_c = torch.where(data_c > args.theta, 1, 0).to(torch.float)  # (n, d)
                # ---------------------------------
                den = torch.mean(data_c)  # (n, d) -> ()
                sim_d = torch.where(torch.sum(data_c, dim=0, keepdim=True) > 0, 1, 0)  # (1, d)
                sim = torch.sum(torch.mean(data_c, dim=0)) / torch.sum(sim_d + 1e-5)
                score = sim / (den + 1e-5)
                # ---------------------------------
                scoresD[label_idx].append(round(den.item(), 2))
                scoresS[label_idx].append(round(sim.item(), 2))
                scores[label_idx].append(round(score.item(), 2))
        # print(scoresD)
        # print('-' * 10)
        # print(scoresS)
        # print('-' * 10)
        # print(scores)
        scores_all.append(scores)
    scores_all = np.asarray(scores_all)  # (types, num_labels, layers)
    scores_all = np.mean(scores_all, axis=1)  # (types, layers)
    print(scores_all.shape)

    fig_path = os.path.join(args.save_dir, 'score.npy')
    np.save(fig_path, scores_all)

    # #################### drawline ####################
    # fig_path = os.path.join(args.save_dir, 'score.png')
    # plt.figure(figsize=(9, 6))
    # df = pd.DataFrame(scores_all).transpose()
    # sns.set(style="whitegrid")
    # sns.lineplot(data=df)
    # # plt.legend(labels=cache_types)
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.clf()
    # plt.close()
    # #################### drawline ####################


if __name__ == '__main__':
    main()
