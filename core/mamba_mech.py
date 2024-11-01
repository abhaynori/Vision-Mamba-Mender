import os

import cv2
import numpy as np
import torch
import argparse
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

import loaders
import models
from utils import file_util
from core.constraints import grad


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='sift dir')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(os.path.join(args.save_dir, 'spatial'))
        # os.makedirs(os.path.join(args.save_dir, 'channel'))

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_dir)
    print('RESULT PATH:', args.save_dir)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------

    sum_start_layer = 0
    # cal_start_layer = 20
    # end_layer = 24
    cal_start_layer = 1
    end_layer = 4
    constraint_layers = [layer for layer in range(cal_start_layer, end_layer)]
    print('constraint_layers:', constraint_layers)

    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes,
                              constraint_layers=constraint_layers)
    # model.load_state_dict(torch.load(args.model_path)['model'])
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(data_dir=args.data_dir,
                                    data_name=args.data_name,
                                    data_type='test',
                                    batch_size=32)

    # ----------------------------------------
    # forward
    # ----------------------------------------

    # channels related
    h_all = [[[] for _ in range(10)] for _ in range(end_layer)]
    qk_all = [[[] for _ in range(10)] for _ in range(end_layer)]
    class_names = sorted([d.name for d in os.scandir(args.data_dir) if d.is_dir()])
    modules = []
    # for layer in range(max(constraint_layers) + 1):
    #     if layer in constraint_layers:
    #         modules.append(HookModule(model.layers[layer]))  # module: Block
    #     else:
    #         modules.append(None)

    cls_pos = 98

    for samples in tqdm(data_loader):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        images = inverse_image(inputs)

        # p_labels = torch.argmax(outputs, dim=1)
        nll_loss = -torch.nn.NLLLoss()(outputs, labels)
        model.zero_grad()

        #############################################
        # attention
        #############################################
        # w = []
        # for layer in constraint_layers:
        #     w_ = grad(nll_loss, model.layers[layer].mixer.out_a)
        #     w_[:, :, (cls_pos + 1):] = grad(nll_loss, model.layers[layer].mixer.out_b).flip([-1])[:, :, (cls_pos + 1):]
        #     w.append(w_)
        # # w = [(grad(nll_loss, model.layers[layer].mixer.out_a) +
        # #       grad(nll_loss, model.layers[layer].mixer.out_b).flip([-1]))
        # #      for layer in constraint_layers]  # (layers, b, d_inner, num_seq)
        # w = torch.stack(w, dim=0)  # (layers, b, d_inner, num_seq)
        # w = torch.relu(w)  # (layers, b, d_inner, 1) !!!!!
        # w = torch.mean(w, dim=3, keepdim=True)  # (layers, b, d_inner, 1)
        # w = mm_norm(w, dim=2)

        # # raw attention
        # a = [model.layers[layer].mixer.xai_b for layer in constraint_layers]
        # a = torch.stack(a, dim=0)  # (depth, b, d_inner, num_seq)
        # a = a.clamp(min=0)
        # print('->', a.shape)
        # # a_flatten = torch.reshape(a, (a.shape[0], -1))
        # # a_max = torch.max(a_flatten, dim=1).values  # (b)
        # # a_min = torch.min(a_flatten, dim=1).values  # (b)
        # # a_max = torch.reshape(a_max, (a.shape[0], 1, 1, 1))
        # # a_min = torch.reshape(a_min, (a.shape[0], 1, 1, 1))
        # # a = ((a - a_min) / (a_max - a_min))  # (depth, b, d_inner, num_seq)
        # # a = w * a  # (layers, b, d_inner, num_seq)
        # a = torch.mean(a, dim=2)  # (depth, b, num_seq)

        # aa = [model.layers[layer].mixer.aa for layer in constraint_layers]
        # aa = torch.stack(aa, dim=0)  # (depth, b, d_inner,d_state, num_seq)
        # aa = torch.relu(aa)
        # aa = torch.mean(aa, dim=2)  # (depth, b ,d_state, num_seq)
        # ab = [model.layers[layer].mixer.ab for layer in constraint_layers]
        # ab = torch.stack(ab, dim=0)  # (depth, b, d_inner,d_state, num_seq)
        # ab = torch.relu(ab)
        # ab = torch.mean(ab, dim=2)  # (depth, b ,d_state, num_seq)
        a = [model.layers[layer].mixer.xai_vector for layer in constraint_layers]
        a = torch.stack(a, dim=0)  # (depth, b, num_seq) | (depth, b, d_state, num_seq)

        # ca = [grad(nll_loss, model.layers[layer].mixer.ca) for layer in constraint_layers]
        # ca = torch.stack(ca, dim=0)  # (depth, b, 1, d_state, seq_num)
        # ca = torch.squeeze(ca)  # (depth, b, d_state, seq_num)
        # ca = torch.relu(ca)
        # cb = [grad(nll_loss, model.layers[layer].mixer.cb) for layer in constraint_layers]
        # cb = torch.stack(cb, dim=0)  # (depth, b, 1, d_state, seq_num)
        # cb = torch.squeeze(cb)  # (depth, b, d_state, seq_num)
        # cb = torch.relu(cb)
        # c = ca + cb.flip(-1)

        # gradients
        g = [grad(nll_loss, modules[layer].outputs[0]) for layer in constraint_layers]
        g = torch.stack(g, dim=0)  # (depth, b, num_seq, embed_dim)
        g = g.clamp(min=0)  # (depth, b, num_seq, embed_dim)
        print('->', g.shape)
        # s = torch.max(s, dim=3).values  # (depth, b, num_seq)
        g = torch.mean(g, dim=3)  # (depth, b, num_seq)
        # g_flatten = torch.reshape(g, (g.shape[0], -1))
        # g_max = torch.max(g_flatten, dim=1).values  # (b)
        # g_min = torch.min(g_flatten, dim=1).values  # (b)
        # g_max = torch.reshape(g_max, (g.shape[0], 1, 1))
        # g_min = torch.reshape(g_min, (g.shape[0], 1, 1))
        # g = ((g - g_min) / (g_max - g_min))  # (depth, b, num_seq)

        # mask
        mask_a = a  # (depth, b, num_seq)
        mask_g = g  # (depth, b, num_seq)
        mask_ag = a * g  # (depth, b, num_seq)
        mask_a0_r = compute_rollout_attention(mask_a, start_layer=0)[:, cls_pos, :]  # (b, num_seqs, num_seqs)
        # mask_a20_r = compute_rollout_attention(mask_a, start_layer=20)[:, cls_pos, :]  # (b, num_seqs, num_seqs)
        mask_ag0_r = compute_rollout_attention(mask_ag, start_layer=0)[:, cls_pos, :]  # (b, num_seqs, num_seqs)
        # mask_ag20_r = compute_rollout_attention(mask_ag, start_layer=20)[:, cls_pos, :]  # (b, num_seqs, num_seqs)
        mask_a0_r = torch.cat([mask_a0_r[:, :cls_pos], mask_a0_r[:, (cls_pos + 1):]], dim=-1)  # (b, num_patches)
        # mask_a20_r = torch.cat([mask_a20_r[:, :cls_pos], mask_a20_r[:, (cls_pos + 1):]], dim=-1)  # (b, num_patches)
        mask_ag0_r = torch.cat([mask_ag0_r[:, :cls_pos], mask_ag0_r[:, (cls_pos + 1):]], dim=-1)  # (b, num_patches)
        # mask_ag20_r = torch.cat([mask_ag20_r[:, :cls_pos], mask_ag20_r[:, (cls_pos + 1):]], dim=-1)  # (b, num_patches)

        mask_a = torch.cat([mask_a[:, :, :cls_pos], mask_a[:, :, (cls_pos + 1):]], dim=-1)  # (depth, b, num_patches)
        mask_g = torch.cat([mask_g[:, :, :cls_pos], mask_g[:, :, (cls_pos + 1):]], dim=-1)  # (depth, b, num_patches)
        mask_ag = torch.cat([mask_ag[:, :, :cls_pos], mask_ag[:, :, (cls_pos + 1):]], dim=-1)  # (depth, b, num_patches)

        # all layers from 'start_layer'
        mask_a_all = torch.mean(mask_a[sum_start_layer:], dim=0)  # (b, num_patches)
        mask_g_all = torch.mean(mask_g[sum_start_layer:], dim=0)  # (b, num_patches)
        mask_ag_all = torch.mean(mask_ag[sum_start_layer:], dim=0)  # (b, num_patches)

        # rollout from 'start_layer':
        # rollout = compute_rollout_attention(all_layer_attentions, start_layer)  # (b, num_seq, num_seq)

        # each input
        for b, l in enumerate(labels):
            # select the corresponding mask for all layers
            b_mask_a_all = mask_a_all[b:b + 1, :]  # (b, num_patches) ->  (1, num_patches)
            b_mask_g_all = mask_g_all[b:b + 1, :]  # (b, num_patches) ->  (1, num_patches)
            b_mask_ag_all = mask_ag_all[b:b + 1, :]  # (b, num_patches) ->  (1, num_patches)
            b_mask_a0_r = mask_a0_r[b:b + 1, :]  # (b, num_patches) ->  (1, num_patches)
            # b_mask_a20_r = mask_a20_r[b:b + 1, :]  # (b, num_patches) ->  (1, num_patches)
            b_mask_ag0_r = mask_ag0_r[b:b + 1, :]  # (b, num_patches) ->  (1, num_patches)
            # b_mask_ag20_r = mask_ag20_r[b:b + 1, :]  # (b, num_patches) ->  (1, num_patches)

            # # 处理rollout
            # p = rollout[i, cls_pos, :].unsqueeze(0)  # (1, num_seq)
            # p = torch.cat([p[:, :cls_pos], p[:, (cls_pos + 1):]], dim=-1)  # (1, num_patches)
            # p = p.clamp(min=0)

            # generate and save spatial heatmap
            img = images[b]
            fig_a_all = spatial_heatmap(img, b_mask_a_all)
            fig_g_all = spatial_heatmap(img, b_mask_g_all)
            fig_ag_all = spatial_heatmap(img, b_mask_ag_all)
            fig_a0_r = spatial_heatmap(img, b_mask_a0_r)
            # fig_a20_r = spatial_heatmap(img, b_mask_a20_r)
            fig_ag0_r = spatial_heatmap(img, b_mask_ag0_r)
            # fig_ag20_r = spatial_heatmap(img, b_mask_ag20_r)
            # save_spatial_heatmaps(figs=[img, fig_a_all, fig_g_all, fig_ag_all],
            #                       save_dir=args.save_dir, class_name=class_names[l], img_name=names[b])
            save_spatial_heatmaps(
                figs=[img, fig_a_all, fig_g_all, fig_ag_all, fig_a0_r, fig_ag0_r],
                save_dir=args.save_dir, class_name=class_names[l], img_name=names[b])

            # each depth
            for i, d in enumerate(constraint_layers):
                # 处理raw attention
                b_mask_a_d = mask_a[i, b:b + 1, :]  # (depth, b, num_patches) -> (1, num_patches)
                b_mask_g_d = mask_g[i, b:b + 1, :]  # (depth, b, num_patches) -> (1, num_patches)
                b_mask_ag_d = mask_ag[i, b:b + 1, :]  # (depth, b, num_patches) -> (1, num_patches)

                # # 处理rollout
                # p = rollout[i, cls_pos, :].unsqueeze(0)  # (1, num_seq)
                # p = torch.cat([p[:, :cls_pos], p[:, (cls_pos + 1):]], dim=-1)  # (1, num_patches)
                # p = p.clamp(min=0)

                # generate and save spatial heatmap
                fig_a_d = spatial_heatmap(img, b_mask_a_d)
                fig_g_d = spatial_heatmap(img, b_mask_g_d)
                fig_ag_d = spatial_heatmap(img, b_mask_ag_d)
                save_spatial_heatmaps(figs=[img, fig_a_d, fig_g_d, fig_ag_d],
                                      save_dir=args.save_dir, class_name=class_names[l], img_name=names[b], suffix=d)

    #     #############################################
    #     # attention
    #     #############################################
    #     h = [model.layers[layer].mixer.h_vector for layer in constraint_layers]
    #     h = torch.stack(h, dim=0)  # (depth, b, d_inner, num_seq)
    #     qk = [model.layers[layer].mixer.qk_vector for layer in constraint_layers]
    #     qk = torch.stack(qk, dim=0)  # (depth, b, d_inner, num_seq)
    #     print('-->', h.shape, qk.shape)
    #
    #     for b, l in enumerate(labels):
    #         for i in range(sum_start_layer, len(model.layers)):
    #             h_cls = h[i, b, :, :]  # (d_inner)
    #             h_cls = torch.mean(h_cls, dim=-1)
    #             qk_cls = qk[i, b, :, :]  # (d_inner)
    #             qk_cls = torch.mean(qk_cls, dim=-1)
    #             h_all[i - sum_start_layer][l].append(h_cls.detach().cpu())
    #             qk_all[i - sum_start_layer][l].append(qk_cls.detach().cpu())
    #
    # for l in range(10):
    #     for i in range(sum_start_layer, len(model.layers)):
    #         h_all_d_l = torch.stack(h_all[i - sum_start_layer][l], dim=0)  # (n, d_inner)
    #         qk_all_d_l = torch.stack(qk_all[i - sum_start_layer][l], dim=0)  # (n, d_inner)
    #         print(h_all_d_l.shape, qk_all_d_l.shape)
    #
    #         class_name = class_names[l]
    #         fig_path = os.path.join(args.save_dir, 'channel', class_name, 'h_' + str(i) + '.png')
    #         save_channel_heatmaps(h_all_d_l, fig_path)
    #         fig_path = os.path.join(args.save_dir, 'channel', class_name, 'qk_' + str(i) + '.png')
    #         save_channel_heatmaps(qk_all_d_l, fig_path)
    #         print(fig_path)
    #         print(fig_path)


def mm_norm(a, dim=-1):
    a_min = torch.min(a, dim=dim, keepdim=True)[0]
    a_max = torch.max(a, dim=dim, keepdim=True)[0]
    a_normalized = (a - a_min) / (a_max - a_min + 1e-5)

    return a_normalized


def compute_rollout_attention(alls, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    # all_layer_matrices: (depth, b, num_seq)
    num_tokens = alls.shape[2]  # num_seq
    batch_size = alls.shape[1]  # b
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(alls[0].device)  # (b, num_seq, num_seq)
    print(eye.shape)
    print(alls[0].unsqueeze(1).shape)

    # (b, 1, num_seq) + (b, num_seq, num_seq) = (b, num_seq, num_seq)
    alls = [alls[i].unsqueeze(1) + eye for i in range(len(alls))]
    # all_layer_matrices[0].shape: (b, num_seq, num_seq)
    matrices_aug = [alls[i] / alls[i].sum(dim=-1, keepdim=True) for i in range(len(alls))]
    # matrices_aug[0].shape: (b, num_seq, num_seq)
    joint_attention = matrices_aug[start_layer]  # (b, num_seq, num_seq)
    for i in range(start_layer + 1, len(matrices_aug)):
        # (b, n, m) @ (b, m, p)= (b, n, p)
        joint_attention = matrices_aug[i].bmm(joint_attention)  # (b, num_seq, num_seq)
    return joint_attention  # (b, num_seq, num_seq)


def save_channel_heatmaps(vals, fig_path, fig_w=None, fig_h=None, annot=False):
    if fig_w is None:
        fig_w = vals.shape[1]
    if fig_h is None:
        fig_h = vals.shape[0]
    path, name = os.path.split(fig_path)
    if not os.path.exists(path):
        os.makedirs(path)

    f, ax = plt.subplots(figsize=(fig_w, fig_h), ncols=1)
    sns.heatmap(vals, ax=ax, annot=annot)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def save_spatial_heatmaps(figs, save_dir, class_name, img_name, suffix=None):
    fname, ext = os.path.splitext(img_name)
    if suffix is not None:
        img_name = fname + '_' + str(suffix) + ext
    fig_path = os.path.join(save_dir, class_name, img_name)
    path, name = os.path.split(fig_path)
    if not os.path.exists(path):
        os.makedirs(path)
    print(fig_path)

    fig_lens = len(figs)
    fig, axs = plt.subplots(1, fig_lens, figsize=(10 * fig_lens, 10))
    for i, fig in enumerate(figs):
        axs[i].imshow(fig)
        axs[i].axis('off')
        axs[i].imshow(fig)
        axs[i].axis('off')

    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def spatial_heatmap(img, mask):
    mask = mask.reshape(1, 1, 14, 14)
    mask = torch.nn.functional.interpolate(mask, scale_factor=16, mode='bilinear')
    mask = mask.reshape(224, 224)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.detach().cpu().numpy()
    # img = img.permute(1, 2, 0).data.cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    vis = show_cam_on_image(img, mask)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def inverse_image(img):
    img = img.squeeze()
    img = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1., 1., 1.]),
    ])(img)
    # img = img.detach().cpu()
    img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
    return img


if __name__ == '__main__':
    main()
