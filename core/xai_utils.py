import torch
import numpy as np


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    # all_layer_matrices: (depth, b, num_seq)
    num_tokens = all_layer_matrices[0].shape[1]  # num_seq
    batch_size = all_layer_matrices[0].shape[0]  # b
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(
        all_layer_matrices[0].device)  # (b, num_seq, num_seq)

    # (b, num_seq) + (b, num_seq, num_seq) = (b, num_seq, num_seq)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices[0].shape: (b, num_seq, num_seq)
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                    for i in range(len(all_layer_matrices))]
    # matrices_aug[0].shape: (b, num_seq, num_seq)
    joint_attention = matrices_aug[start_layer]  # (b, num_seq, num_seq)
    for i in range(start_layer + 1, len(matrices_aug)):
        # (b, n, m) @ (b, m, p)= (b, n, p)
        joint_attention = matrices_aug[i].bmm(joint_attention)  # (b, num_seq, num_seq)
    return joint_attention  # (b, num_seq, num_seq)


def generate_raw_attn(model, image, start_layer=15):
    image.requires_grad_()
    logits = model(image)
    all_layer_attentions = []
    cls_pos = 98
    for layeridx in range(len(model.layers)):  # depth:24
        attn_heads = model.layers[layeridx].mixer.xai_vector  # (b, d_inner, num_seq)
        # attn_heads = model.layers[layeridx].mixer.h_vector  # (b, d_inner, num_seq)
        # attn_heads = model.layers[layeridx].mixer.qk_vector  # (b, d_inner, num_seq)
        attn_heads = (attn_heads - attn_heads.min()) / (attn_heads.max() - attn_heads.min())  # (b, d_inner, num_seq)
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()  # (b, num_seq)
        all_layer_attentions.append(avg_heads)  # (depth, b, num_seq)
    # torch.cat(all_layer_attentions[start_layer:], dim=0): (depth-start_layer, num_seq)
    p = torch.cat(all_layer_attentions[start_layer:], dim=0).mean(dim=0).unsqueeze(0)  # (1, num_seq)
    p = torch.cat([p[:, :cls_pos], p[:, (cls_pos + 1):]], dim=-1)  # (1, num_patches)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits  # (1, num_patches)


def generate_rollout(model, image, start_layer=15):
    image.requires_grad_()
    logits = model(image)
    all_layer_attentions = []
    cls_pos = 98
    for layer in range(len(model.layers)):
        attn_heads = model.layers[layer].mixer.xai_vector  # (b, d_inner, num_seq)
        # attn_heads = model.layers[layer].mixer.h_vector  # (b, d_inner, num_seq)
        # attn_heads = model.layers[layer].mixer.qk_vector  # (b, d_inner, num_seq)
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()  # (b, num_seq)
        all_layer_attentions.append(avg_heads)  # (depth, b, num_seq)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)  # (b, num_seq, num_seq)
    p = rollout[0, cls_pos, :].unsqueeze(0)  # (1, num_seq)
    p = torch.cat([p[:, :cls_pos], p[:, (cls_pos + 1):]], dim=-1)  # (1, num_patches)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits  # (1, num_patches)


def generate_mamba_attr(model, image, start_layer=15):
    image.requires_grad_()
    logits = model(image)
    index = np.argmax(logits.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    all_layer_attentions = []
    cls_pos = 98
    for layeridx in range(len(model.layers)):
        attn_heads = model.layers[layeridx].mixer.xai_vector.clamp(min=0)  # (b, d_inner, num_seq)
        # attn_heads = model.layers[layeridx].mixer.h_vector.clamp(min=0)  # (b, d_inner, num_seq)
        # attn_heads = model.layers[layeridx].mixer.qk_vector.clamp(min=0)  # (b, d_inner, num_seq)
        s = model.layers[
            layeridx].get_gradients().squeeze().detach()  # [1:, :].clamp(min=0).max(dim=1)[0].unsqueeze(0) # (num_seq, embed_dim)
        s = s.clamp(min=0).max(dim=1)[0].unsqueeze(0)  # (1, num_seq)
        s = (s - s.min()) / (s.max() - s.min())  # (1, num_seq)
        attn_heads = (attn_heads - attn_heads.min()) / (attn_heads.max() - attn_heads.min())  # (b, d_inner, num_seq)
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()  # (b, num_seq)
        fused = avg_heads * s  # (1, num_seq)
        all_layer_attentions.append(fused)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer)
    p = rollout[0, cls_pos, :].unsqueeze(0)
    p = torch.cat([p[:, :cls_pos], p[:, (cls_pos + 1):]], dim=-1)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits
