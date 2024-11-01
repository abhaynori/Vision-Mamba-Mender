import argparse
import os

import torch
import numpy as np
from PIL import Image

from state_external_visualize import load_names, load_state_external


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_layers', default='', type=int, help='num layers')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--cache_types', default=None, nargs='+', type=str, help='cache types')
    parser.add_argument('--sample_dir', default='', type=str, help='sample dir')
    parser.add_argument('--sample_name_path', default='', type=str, help='sample name path')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    layers = [l for l in range(args.num_layers)]  # TODO Not necessarily all layers
    labels = [c for c in range(args.num_classes)]
    cls_pos = 98  # TODO Are there classification tokens?

    sample_names, class_names = load_names(args.sample_dir, args.sample_name_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for cache_type in args.cache_types:
        for layer in layers:

            img_dir = os.path.join(args.save_dir, cache_type + '-ag', 'layer{}'.format(layer))
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            #################### data ####################
            print('===> load data for layer', layer)
            data = load_state_external(args.data_dir, layer, cache_type)
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
                    name, _ = os.path.splitext(sample_names[label][sample_idx])
                    img_path = os.path.join(img_dir, name + '.png')
                    gray.save(img_path)
                #################### cam ####################


if __name__ == '__main__':
    main()
