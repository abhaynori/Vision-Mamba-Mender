import numpy as np
import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import pickle

import loaders
import models


class HookModule:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.outputs = None
        module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs[0]
        self.outputs = outputs


def grads(outputs, inputs, retain_graph=True, create_graph=False):
    return torch.autograd.grad(outputs=outputs,
                               inputs=inputs,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]


class FeatureSift:
    def __init__(self, model, model_name, cache_layers, num_classes, num_samples, cache_type='x0', value_type='a',
                 is_high_confidence=True):
        # self.modules = [HookModule(module) for module in modules]  # [L, C, N, [channels]]
        self.model = model
        self.cache_layers = cache_layers
        self.num_samples = num_samples
        self.cache_type = cache_type
        self.value_type = value_type
        self.is_high_confidence = is_high_confidence

        # (l(layer), c(class), n(samples) * [d(dims), s(seqs)])
        self.values = [[[] for _ in range(num_classes)] for _ in range(len(cache_layers))]
        self.scores = torch.zeros((len(cache_layers), num_classes, num_samples))  # (l, c, n)
        self.nums = torch.zeros((len(cache_layers), num_classes), dtype=torch.long)  # (l, C)
        self.names = [[None for _ in range(num_samples)] for _ in range(num_classes)]  # (c, n)

        self.model_name = model_name
        if 'vim' in model_name:  # vim, local_vim
            self.mamba_name = 'vim'
        elif 'vmamba' in model_name:  # vmamba, efficient-vmamba
            self.mamba_name = 'vmamba'
        elif 'simba' in model_name:  # simba
            self.mamba_name = 'simba'

    def __call__(self, outputs, labels, names, layers=None):
        softmaxs = nn.Softmax(dim=1)(outputs.detach())  # (b, c)

        for layer_i, layer in enumerate(self.cache_layers):
            # pre processing
            if self.mamba_name == 'vim':  # vim, local_vim
                values = self.model.layers[layer].mixer.cache[self.cache_type]
            elif self.mamba_name == 'vmamba':  # vmamba, efficient-vmamba
                if 'efficient' in self.model_name:
                    layer = 1
                else:
                    layer = 3
                depth = 1
                values = self.model.layers[layer][0][depth].op.cache[self.cache_type]
            elif self.mamba_name == 'simba':  # simba
                # depth = 1
                # values = self.model.block4[depth].attn.mamba.cache[self.cache_type]
                values = self.model.post_network[0].attn.mamba.cache[self.cache_type]
            assert values is not None

            # a or g
            if 'g' == self.value_type:
                nll_loss = -nn.NLLLoss()(outputs, labels)
                values = grads(nll_loss, values)

            # post processing
            if self.mamba_name == 'vim':  # vim, local_vim
                values = values
            elif self.mamba_name == 'vmamba':  # vmamba, efficient-vmamba
                if 'x' in self.cache_type:
                    values = values.reshape(values.size(0), values.size(1), -1)
                elif 'c' in self.cache_type:
                    values = values.reshape(values.size(0), values.size(1), -1)
                elif 's' in self.cache_type:
                    values = values.permute(0, 3, 1, 2).contiguous()
                    values = values.reshape(values.size(0), values.size(1), -1)
                elif 'z' in self.cache_type:
                    values = values.permute(0, 3, 1, 2).contiguous()
                    values = values.reshape(values.size(0), values.size(1), -1)
            elif self.mamba_name == 'simba':  # simba
                values = values.permute(0, 2, 1).contiguous()

            values = values.detach().cpu().numpy()

            for i, label in enumerate(labels):  # each datas
                # w/o selection
                score = softmaxs[i][label]
                self.values[layer_i][label].append(values[i])
                self.scores[layer_i][label][self.nums[layer_i][label]] = score
                self.names[label][self.nums[layer_i][label]] = names[i]
                self.nums[layer_i][label] += 1

                # w/ selection
                # score = softmaxs[i][label]  # (b, c) -> ()
                # if self.is_high_confidence:  # sift high confidence
                #     if self.nums[layer][label] == self.num_samples:
                #         score_min, index = torch.min(self.scores[layer][label], dim=0)
                #         if score > score_min:
                #             self.values[layer][label][index] = values[i]
                #             self.scores[layer][label][index] = score
                #     else:
                #         self.values[layer][label].append(values[i])
                #         self.scores[layer][label][self.nums[layer][label]] = score
                #         self.nums[layer][label] += 1
                # else:  # sift low confidence
                #     if self.nums[layer][label] == self.num_samples:
                #         score_max, index = torch.max(self.scores[layer][label], dim=0)
                #         if score < score_max:
                #             self.values[layer][label][index] = values[i]
                #             self.scores[layer][label][index] = score
                #     else:
                #         self.values[layer][label].append(values[i])
                #         self.scores[layer][label][self.nums[layer][label]] = score
                #         self.nums[layer][label] += 1

    def save(self, save_dir, save_name):  # (l, c, n, d, s)
        # print(self.nums)
        # print(self.scores)

        for layer_i, layer in enumerate(self.cache_layers):
            values = self.values[layer_i]  # (c, n, d, s)
            values = np.asarray(values)
            # print(values.shape)

            save_path = os.path.join(save_dir, 'layer{}_{}.pkl'.format(layer, save_name))
            values_file = open(save_path, 'wb')
            pickle.dump(values, values_file)

        # print(self.names)
        save_path = os.path.join(save_dir, 'sample_names.pkl'.format(save_name))
        values_file = open(save_path, 'wb')
        pickle.dump(self.names, values_file)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--num_samples', default=10, type=int, help='num samples')
    parser.add_argument('--cache_layers', default=[], nargs='*', type=int, help='cache layers')
    parser.add_argument('--cache_type', default='', type=str, help='cache type')
    parser.add_argument('--value_type', default='', type=str, help='value type')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    assert args.value_type in ['a', 'g']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_dir)
    print('SAVE PATH:', args.save_dir)
    print('SAVE NAME:', args.cache_layers, args.cache_type, args.value_type)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name,
                              num_classes=args.num_classes,
                              constraint_layers=args.cache_layers)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(data_dir=args.data_dir, data_name=args.data_name, data_type='test')

    feature_sift = FeatureSift(model=model,
                               model_name=args.model_name,
                               cache_layers=args.cache_layers,
                               num_classes=args.num_classes,
                               num_samples=args.num_samples,
                               cache_type=args.cache_type,
                               value_type=args.value_type,
                               is_high_confidence=True)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for samples in tqdm(data_loader):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        if 'g' == args.value_type:
            outputs = model(inputs)
        else:
            with torch.no_grad():
                outputs = model(inputs)

        feature_sift(outputs=outputs, labels=labels, names=names, layers=None)

        if 'g' == args.value_type:  # release cache
            loss = torch.sum(outputs)
            loss.backward()

    feature_sift.save(save_dir=args.save_dir, save_name=args.cache_type + '-' + args.value_type)


if __name__ == '__main__':
    main()
