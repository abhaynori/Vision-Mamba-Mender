import os
import torch
from torchvision import transforms


def grad(outputs, inputs, retain_graph=True, create_graph=True):
    return torch.autograd.grad(outputs=outputs,
                               inputs=inputs,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]


class StateConstraint:
    def __init__(self, model, model_name, alpha, beta, external_cache_layers, internal_cache_layers,
                 external_cache_types, internal_cache_types, internal_mask_dir=None):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.external_cache_layers = external_cache_layers  # []
        self.internal_cache_layers = internal_cache_layers  # []
        self.external_cache_types = external_cache_types  # []
        self.internal_cache_types = internal_cache_types  # []

        self.model_name = model_name
        if 'vim' in model_name:  # vim, local_vim
            self.mamba_name = 'vim'
        elif 'vmamba' in model_name:  # vmamba, efficient-vmamba
            self.mamba_name = 'vmamba'
        elif 'simba' in model_name:  # simba
            self.mamba_name = 'simba'

        if self.mamba_name == 'vim':  # vim, local_vim
            self.cls_pos = 98
            self.branches = ['0', '1']
        elif self.mamba_name == 'vmamba':  # vmamba, efficient-vmamba
            self.cls_pos = None
            self.branches = ['0']
        elif self.mamba_name == 'simba':  # simba
            self.cls_pos = 0
            self.branches = ['0']
        print('=====> current mamba type:', self.mamba_name)

        # load internal mask
        if len(self.internal_cache_layers) != 0:  # w/o internal constraint
            self.internal_masks = {}  # {'x0':(l, c, d),...}
            for cache_type in self.internal_cache_types:
                if cache_type is not 'h':
                    for branch in self.branches:
                        internal_masks = []
                        for layer in self.internal_cache_layers:
                            mask_path = os.path.join(internal_mask_dir,
                                                     'layer{}_{}-ag.pt'.format(layer, cache_type + branch))
                            internal_masks.append(torch.load(mask_path))  # (c, d) -> (l, c, d)
                        self.internal_masks[cache_type + branch] = torch.stack(internal_masks, dim=0).cuda()
                else:
                    self.branches = ['']
                    internal_masks = []
                    for layer in self.internal_cache_layers:
                        mask_path = os.path.join(internal_mask_dir,
                                                 'layer{}_{}-ag.pt'.format(layer, cache_type))
                        internal_masks.append(torch.load(mask_path))  # (c, d) -> (l, c, d)
                    self.internal_masks[cache_type] = torch.stack(internal_masks, dim=0).cuda()
        else:
            if len(self.external_cache_layers) == 0:
                print('=====> no cache')

            # # new for h
            # for cache_type in self.internal_cache_types:
            #     internal_masks = []
            #     for layer in self.internal_cache_layers:
            #         mask_path = os.path.join(internal_mask_dir,
            #                                  'layer{}_{}-ag.pt'.format(layer, cache_type))
            #         internal_masks.append(torch.load(mask_path))  # (c, d) -> (l, c, d)
            #     self.internal_masks[cache_type] = torch.stack(internal_masks, dim=0).cuda()

    def loss_external(self, outputs, labels, masks):
        if len(self.external_cache_layers) == 0:
            return torch.tensor(0.0).to(labels.device)

        # --------- load state value ---------
        nll_loss = -torch.nn.NLLLoss()(outputs, labels)
        sta_ext = []
        if self.mamba_name == 'simba':  # simba
            for cache_type in self.external_cache_types:
                sta_ext.append(self.load_state_external_simba(cache_type, nll_loss))  # (l, b, d, s)
            sta_ext = torch.stack(sta_ext)  # (types, l, b, d, s)
            sta_ext = torch.sum(sta_ext, dim=0)  # (l, b, d, s)
            sta_ext = torch.mean(sta_ext, dim=(0, 2))  # (l, b, d, s) -> (b, s)

            # --------- reshape state ---------
            sta_ext = torch.reshape(sta_ext, (sta_ext.shape[0], 1, 7, 7))  # (b, 1, 14, 14)
            sta_ext = torch.nn.functional.interpolate(sta_ext, scale_factor=32, mode='bilinear')  # (b, 1, 224, 224)
        elif self.mamba_name == 'vmamba':  # vim, local_vim
            for cache_type in self.external_cache_types:
                sta_ext.append(self.load_state_external_vmamba(cache_type, nll_loss))  # (l, b, d, s)
            sta_ext = torch.stack(sta_ext)  # (types, l, b, d, s)
            sta_ext = torch.sum(sta_ext, dim=0)  # (l, b, d, s)
            sta_ext = torch.mean(sta_ext, dim=(0, 2))  # (l, b, d, s) -> (b, s)

            # --------- reshape state ---------
            sta_ext = torch.reshape(sta_ext, (sta_ext.shape[0], 1, 14, 14))  # (b, 1, 14, 14)
            sta_ext = torch.nn.functional.interpolate(sta_ext, scale_factor=16, mode='bilinear')  # (b, 1, 224, 224)
        else:  # vim, local_vim
            for cache_type in self.external_cache_types:
                sta_ext.append(self.load_state_external(cache_type, nll_loss))  # (l, b, d, s)
            sta_ext = torch.stack(sta_ext)  # (types, l, b, d, s)
            sta_ext = torch.sum(sta_ext, dim=0)  # (l, b, d, s)
            sta_ext = torch.mean(sta_ext, dim=(0, 2))  # (l, b, d, s) -> (b, s)

            # --------- kick out cls state and reshape state ---------
            sta_ext = torch.cat([sta_ext[:, :self.cls_pos], sta_ext[:, (self.cls_pos + 1):]], dim=-1)  # (b, s)
            sta_ext = torch.reshape(sta_ext, (sta_ext.shape[0], 1, 14, 14))  # (b, 1, 14, 14)
            sta_ext = torch.nn.functional.interpolate(sta_ext, scale_factor=16, mode='bilinear')  # (b, 1, 224, 224)

        # --------- constraint with mask ---------
        masks = masks.to(device=sta_ext.device, dtype=sta_ext.dtype)  # (b, 1, 224, 224)
        # masks = transforms.Resize((sta_ext.shape[2], sta_ext.shape[3]))(masks)  # (b, 1, 14, 14)
        masks = 1 - masks
        # print(torch.min(masks), torch.max(masks), torch.mean(masks))

        loss = torch.sum((sta_ext * masks)) / sta_ext.shape[0]
        return loss * self.alpha

    def loss_internal(self, outputs, labels):
        if len(self.internal_cache_layers) == 0:
            return torch.tensor(0.0).to(labels.device)

        # --------- load state value ---------
        nll_loss = -torch.nn.NLLLoss()(outputs, labels)

        loss = torch.tensor(0.0).to(labels.device)
        for cache_type in self.internal_cache_types:
            for branch in self.branches:
                if self.mamba_name == 'simba':  # simba
                    # --------- load state value ---------
                    sta_int = self.load_state_internal_simba(cache_type + branch, nll_loss)  # (l, b, d, s)
                    # --------- pick out cls state ---------
                    sta_int = sta_int[:, :, :, self.cls_pos]  # (l, b, d)
                    # --------- constraint with mask ---------
                    mask = torch.index_select(self.internal_masks[cache_type + branch],
                                              index=labels, dim=1)  # (l, c, d) -> (l, b, d)
                    mask = 1 - mask
                    loss = loss + torch.mean(sta_int * mask)
                elif self.mamba_name == 'vmamba':  # vim, local_vim
                    # --------- load state value ---------
                    sta_int = self.load_state_internal_vmamba(cache_type + branch, nll_loss)  # (l, b, d, s)
                    # --------- pick out cls state ---------
                    sta_int = torch.mean(sta_int, dim=3)  # (l, b, d, s) -> (l, b, d)
                    # --------- constraint with mask ---------
                    mask = torch.index_select(self.internal_masks[cache_type + branch],
                                              index=labels, dim=1)  # (l, c, d) -> (l, b, d)
                    mask = 1 - mask
                    loss = loss + torch.mean(sta_int * mask)
                else:  # vim, local_vim
                    # --------- load state value ---------
                    sta_int = self.load_state_internal(cache_type + branch, nll_loss)  # (l, b, d, s)
                    # --------- pick out cls state ---------
                    sta_int = sta_int[:, :, :, self.cls_pos]  # (l, b, d)
                    # --------- constraint with mask ---------
                    mask = torch.index_select(self.internal_masks[cache_type + branch],
                                              index=labels, dim=1)  # (l, c, d) -> (l, b, d)
                    mask = 1 - mask
                    loss = loss + torch.mean(sta_int * mask)
            # # new for h
            # # --------- load state value ---------
            # sta_int = self.load_state_internal(cache_type, nll_loss)  # (l, b, s, d)
            # sta_int = sta_int.permute(0, 1, 3, 2)  # (l, b, s, d) -> (l, b, d, s)
            # # --------- pick out cls state ---------
            # sta_int = sta_int[:, :, :, self.cls_pos]  # (l, b, d)
            # # --------- constraint with mask ---------
            # mask = torch.index_select(self.internal_masks[cache_type],
            #                           index=labels, dim=1)  # (l, c, d) -> (l, b, d)
            # mask = 1 - mask
            # loss = loss + torch.mean(sta_int * mask)

        return loss * self.beta

    def load_state_external_simba(self, cache_type, nll_loss):
        depth = 1
        cache = self.model.block4[depth].attn.mamba.cache
        # cache = self.model.post_network[0].attn.mamba.cache
        # print(cache.keys())
        # print(cache['x0'].shape)
        # print(cache['c0'].shape)
        # print(cache['s0'].shape)
        # print(cache['z0'].shape)

        # --------- activation ---------
        a = [cache[cache_type + '0']]
        a = torch.stack(a, dim=0)  # (l, b, d, s)
        a = torch.relu(a)  # (l, b, d, s)

        # --------- gradient ---------
        g = [grad(nll_loss, cache[cache_type + '0'])]
        g = torch.stack(g, dim=0)  # (l, b, d, s)
        g = torch.relu(g)  # (l, b, d, s)

        g = torch.mean(g, dim=3, keepdim=True)  # (l, b, d, 1)

        return a * g  # (l, b, d, s)

    def load_state_external_vmamba(self, cache_type, nll_loss):
        # [2,27,2,2]
        # layer = 3
        # depth = 1
        layer = 0
        depth = 1
        cache = self.model.layers[layer][0][depth].op.cache
        # print(cache.keys())
        # print(cache['x0'].shape)
        # print(cache['c0'].shape)
        # print(cache['s0'].shape)
        # print(cache['z0'].shape)

        # --------- activation ---------
        value = cache[cache_type + '0']
        if cache_type == 'x':
            value = value.reshape(value.size(0), value.size(1), -1)
        elif cache_type == 'c':
            value = value.reshape(value.size(0), value.size(1), -1)
        elif cache_type == 's':
            value = value.permute(0, 3, 1, 2).contiguous()
            value = value.reshape(value.size(0), value.size(1), -1)
        elif cache_type == 'z':
            value = value.permute(0, 3, 1, 2).contiguous()
            value = value.reshape(value.size(0), value.size(1), -1)
        # print('value.shape', value.shape)
        a = [value]
        a = torch.stack(a, dim=0)  # (l, b, d, s)
        a = torch.relu(a)  # (l, b, d, s)

        # --------- gradient ---------
        value = grad(nll_loss, cache[cache_type + '0']).detach()  # the only detach
        if cache_type == 'x':
            value = value.reshape(value.size(0), value.size(1), -1)
        elif cache_type == 'c':
            value = value.reshape(value.size(0), value.size(1), -1)
        elif cache_type == 's':
            value = value.permute(0, 3, 1, 2).contiguous()
            value = value.reshape(value.size(0), value.size(1), -1)
        elif cache_type == 'z':
            value = value.permute(0, 3, 1, 2).contiguous()
            value = value.reshape(value.size(0), value.size(1), -1)
        # print('value.shape', value.shape)
        g = [value]
        g = torch.stack(g, dim=0)  # (l, b, d, s)
        g = torch.relu(g)  # (l, b, d, s)

        g = torch.mean(g, dim=3, keepdim=True)  # (l, b, d, 1)

        return a * g  # (l, b, d, s)

    # vision mamba (ViM), local ViM
    def load_state_external(self, cache_type, nll_loss):  # w/ branch // TODO Specific to the model
        if cache_type is not 'h':
            # --------- activation ---------
            a0 = [self.model.layers[layer].mixer.cache[cache_type + '0']
                  for layer in self.external_cache_layers]
            a0 = torch.stack(a0, dim=0)  # (l, b, d, s)
            a1 = [self.model.layers[layer].mixer.cache[cache_type + '1'].flip(-1)
                  for layer in self.external_cache_layers]
            a1 = torch.stack(a1, dim=0)  # (l, b, d, s)
            a = a0 + a1  # (l, b, d, s)
            a = torch.relu(a)  # (l, b, d, s)

            # --------- gradient ---------
            g0 = [grad(nll_loss, self.model.layers[layer].mixer.cache[cache_type + '0'])
                  for layer in self.external_cache_layers]
            g0 = torch.stack(g0, dim=0)  # (l, b, d, s)
            g1 = [grad(nll_loss, self.model.layers[layer].mixer.cache[cache_type + '1']).flip(-1)
                  for layer in self.external_cache_layers]
            g1 = torch.stack(g1, dim=0)  # (l, b, d, s)
            g = g0 + g1  # (l, b, d, s)
            g = torch.relu(g)  # (l, b, d, s)

            g = torch.mean(g, dim=3, keepdim=True)  # (l, b, d, 1)
        else:
            # --------- activation ---------
            a0 = [self.model.layers[layer].mixer.cache[cache_type]
                  for layer in self.external_cache_layers]
            a0 = torch.stack(a0, dim=0)  # (l, b, d, s)
            a = torch.permute(a0, (0, 1, 3, 2))  # (l, b, s, d) -> (l, b, d, s)
            a = torch.relu(a)  # (l, b, d, s)

            # --------- gradient ---------
            g0 = [grad(nll_loss, self.model.layers[layer].mixer.cache[cache_type])
                  for layer in self.external_cache_layers]
            g0 = torch.stack(g0, dim=0)  # (l, b, d, s)
            g = torch.permute(g0, (0, 1, 3, 2))  # (l, b, s, d) -> (l, b, d, s)
            g = torch.relu(g)  # (l, b, d, s)

            g = torch.mean(g, dim=3, keepdim=True)  # (l, b, d, 1)

        return a * g  # (l, b, d, s)

    def load_state_internal_simba(self, cache_type, nll_loss):
        cache = self.model.post_network[0].attn.mamba.cache
        # print(cache.keys())
        # print(cache['x0'].shape)
        # print(cache['c0'].shape)
        # print(cache['s0'].shape)
        # print(cache['z0'].shape)

        # --------- activation ---------
        value = cache[cache_type]
        value = value.permute(0, 2, 1).contiguous()
        a = [value]
        a = torch.stack(a, dim=0)  # (l, b, d, s)
        a = torch.relu(a)  # (l, b, d, s)

        # --------- gradient ---------
        value = grad(nll_loss, cache[cache_type])
        value = value.permute(0, 2, 1).contiguous()
        g = [value]
        g = torch.stack(g, dim=0)  # (l, b, d, s)
        g = torch.relu(g)  # (l, b, d, s)

        g = torch.mean(g, dim=3, keepdim=True)  # (l, b, d, 1)

        return a * g  # (l, b, d, s)

    def load_state_internal_vmamba(self, cache_type, nll_loss):  # w/o branch
        if 'efficient' in self.model_name:
            layer = 1
        else:
            layer = 3
        depth = 1
        cache = self.model.layers[layer][0][depth].op.cache

        # --------- activation ---------
        value = cache[cache_type]
        if 'x' in cache_type:
            value = value.reshape(value.size(0), value.size(1), -1)
        elif 'c' in cache_type:
            value = value.reshape(value.size(0), value.size(1), -1)
        elif 's' in cache_type:
            value = value.permute(0, 3, 1, 2).contiguous()
            value = value.reshape(value.size(0), value.size(1), -1)
        elif 'z' in cache_type:
            value = value.permute(0, 3, 1, 2).contiguous()
            value = value.reshape(value.size(0), value.size(1), -1)
        # print('value.shape', value.shape)
        a = [value]
        a = torch.stack(a, dim=0)  # (l, b, d, s)
        a = torch.relu(a)  # (l, b, d, s)

        # --------- gradient ---------
        value = grad(nll_loss, cache[cache_type])
        if 'x' in cache_type:
            value = value.reshape(value.size(0), value.size(1), -1)
        elif 'c' in cache_type:
            value = value.reshape(value.size(0), value.size(1), -1)
        elif 's' in cache_type:
            value = value.permute(0, 3, 1, 2).contiguous()
            value = value.reshape(value.size(0), value.size(1), -1)
        elif 'z' in cache_type:
            value = value.permute(0, 3, 1, 2).contiguous()
            value = value.reshape(value.size(0), value.size(1), -1)
        # print('value.shape', value.shape)
        g = [value]
        g = torch.stack(g, dim=0)  # (l, b, d, s)
        g = torch.relu(g)  # (l, b, d, s)

        return a * g  # (l, b, d, s)

    def load_state_internal(self, cache_type, nll_loss):  # w/o branch // TODO Specific to the model
        if cache_type is not 'h':
            # --------- activation ---------
            a = [self.model.layers[layer].mixer.cache[cache_type]
                 for layer in self.internal_cache_layers]
            a = torch.stack(a, dim=0)  # (l, b, d, s)
            a = torch.relu(a)  # (l, b, d, s)

            # --------- gradient ---------
            g = [grad(nll_loss, self.model.layers[layer].mixer.cache[cache_type])
                 for layer in self.internal_cache_layers]
            g = torch.stack(g, dim=0)  # (l, b, d, s)
            g = torch.relu(g)  # (l, b, d, s)
        else:
            # --------- activation ---------
            a = [self.model.layers[layer].mixer.cache[cache_type]
                 for layer in self.internal_cache_layers]
            a = torch.stack(a, dim=0)  # (l, b, d, s)
            a = torch.permute(a, (0, 1, 3, 2))  # (l, b, s, d) -> (l, b, d, s)
            a = torch.relu(a)  # (l, b, d, s)

            # --------- gradient ---------
            g = [grad(nll_loss, self.model.layers[layer].mixer.cache[cache_type])
                 for layer in self.internal_cache_layers]
            g = torch.stack(g, dim=0)  # (l, b, d, s)
            g = torch.permute(g, (0, 1, 3, 2))  # (l, b, s, d) -> (l, b, d, s)
            g = torch.relu(g)  # (l, b, d, s)

        return a * g  # (l, b, d, s)

    def del_cache(self):
        if self.mamba_name == 'simba':
            if 9 in self.external_cache_layers or 9 in self.internal_cache_layers:
                depth = 1
                del self.model.block4[depth].attn.mamba.cache
            elif 10 in self.external_cache_layers or 10 in self.internal_cache_layers:
                del self.model.post_network[0].attn.mamba.cache
        elif self.mamba_name == 'vmamba':
            layers = []
            if 1 in self.external_cache_layers or 1 in self.internal_cache_layers:
                layers.append(0)
            elif 3 in self.external_cache_layers or 3 in self.internal_cache_layers:
                layers.append(1)
            elif 14 in self.external_cache_layers or 14 in self.internal_cache_layers:
                layers.append(3)
            depth = 1
            for layer in layers:
                del self.model.layers[layer][0][depth].op.cache
        else:
            for layer in list(set(self.external_cache_layers + self.internal_cache_layers)):
                del self.model.layers[layer].mixer.cache
