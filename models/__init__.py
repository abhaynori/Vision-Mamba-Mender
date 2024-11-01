import torch
import random
import numpy as np

from models.vision_mamba.mamba import \
    vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as vim_tiny
from models.vision_mamba.mamba import \
    vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as vim_tiny_s
from models.vision_mamba.mamba import \
    vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as vim_small
from models.vision_mamba.mamba import \
    vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as vim_small_
from models.vision_mamba.mamba import \
    vim_cifar
from models.vision_mamba.mamba import \
    vim_tiny_d12
from models.vision_mamba_x.mamba import \
    vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_x as vim_tiny_x
from models.vision_mamba_x.mamba import \
    vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_x as vim_tiny_s_x
from models.vision_mamba_x.mamba import \
    vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_x as vim_small_x
from models.vision_mamba_x.mamba import \
    vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_x as vim_small_s_x
from models.vision_mamba_x.mamba import \
    vim_cifar_x
from models.vision_mamba_x.mamba import \
    vim_tiny_d12_x
# from models.simba_x.simba_x import \
#     simba_s, simba_b, simba_l
# from models.simba_x.simba_x import simba_s as simba_s_x
# from models.mamba_nd.mamba import Mamba2DModel as \
#     mamba_nd
# from models.mamba_nd_x.mamba import Mamba2DModel as \
#     mamba_nd_x
# from models.efficientvssm.models.vmamba_efficient import EfficientVSSM as \
#     efficientvssm
# from models.efficientvssm_x.models.vmamba_efficient import EfficientVSSM as \
#     efficientvssm_x
# from models.vmamba.vmamba import VSSM  as \
#     vmamba
# from models.vmamba_x.vmamba import VSSM  as \
#     vmamba_x
# from models.local_vim.local_vim import local_vim_tiny
# from models.local_vim_x.local_vim import local_vim_tiny as \
#     local_vim_tiny_x
# from models.plain_mamba.plain_mamba.plain_mamba import PlainMamba as \
#     plain_mamba
# from models.plain_mamba_x.plain_mamba.plain_mamba import PlainMamba as \
#     plain_mamba_x
from models.local_mamba.model import local_vim_tiny_middle_cls_token as local_vim_tiny
from models.local_mamba_x.model import local_vim_tiny_middle_cls_token as local_vim_tiny_x
from models.simba.simba import simba_s_new as simba_s
from models.simba_x.simba import simba_s_new as simba_s_x
from models.vmamba.vmamba import vmamba_t as vmamba_t
from models.vmamba_x.vmamba import vmamba_t as vmamba_t_x
from models.efficient_vamamba.vmamba_efficient import efficientvmamba_t as efficientvmamba_t
from models.efficient_vamamba_x.vmamba_efficient import efficientvmamba_t as efficientvmamba_t_x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_models(model_name, in_channels=3, num_classes=10, constraint_layers=None):
    print('-' * 50)
    print('MODEL NAME:', model_name)
    print('NUM CLASSES:', num_classes)
    print('-' * 50)

    model = None
    if model_name == 'vim_tiny':
        model = vim_tiny(num_classes=num_classes)
    if model_name == 'vim_tiny_x':
        model = vim_tiny_x(num_classes=num_classes, constraint_layers=constraint_layers)
    if model_name == 'vim_d12':
        model = vim_tiny_d12(num_classes=num_classes)
    if model_name == 'vim_d12_x':
        model = vim_tiny_d12_x(num_classes=num_classes, constraint_layers=constraint_layers)
    # if model_name == 'vim_tiny_s':
    #     model = vim_tiny_s()
    # if model_name == 'vim_tiny_s_x':
    #     model = vim_tiny_s_x()
    # if model_name == 'vim_cifar':
    #     model = vim_cifar()
    # if model_name == 'vim_cifar_x':
    #     model = vim_cifar_x()
    # if model_name == 'simba_s':
    #     model = simba_s(num_classes=num_classes, drop_path=0.1, token_label=False)
    # if model_name == 'simba_s_x':
    #     model = simba_s_x(num_classes=num_classes, drop_path=0.1, token_label=False, constraint_layers=constraint_layers, bimamba_type='single')
    # if model_name == 'simba_b':
    #     model = simba_b(num_classes=num_classes, drop_path=0.1, token_label=False)
    # if model_name == 'simba_l':
    #     model = simba_l(num_classes=num_classes, drop_path=0.3, token_label=False)
    # if model_name == 'mamba_nd_s':
    #     model = mamba_nd(arch='small',img_size=224,patch_size=8,out_type='avg_featmap',drop_path_rate=0.1,drop_rate=0.1,with_cls_token=False,final_norm=True,fused_add_norm=False,d_state=16,is_2d=False,use_v2=False,use_nd=False,constant_dim=True,downsample=(9,),force_a2=False,use_mlp=False,num_classes=num_classes)
    # if model_name == 'mamba_nd_s_x':
    #     model = mamba_nd_x(arch='small',img_size=224,patch_size=8,out_type='avg_featmap',drop_path_rate=0.1,drop_rate=0.1,with_cls_token=False,final_norm=True,fused_add_norm=False,d_state=16,is_2d=False,use_v2=False,use_nd=False,constant_dim=True,downsample=(9,),force_a2=False,use_mlp=False,num_classes=num_classes, constraint_layers=constraint_layers)
    # if model_name == 'vmamba_s':
    #     model = vmamba(patch_size=4, in_chans=3, num_classes=num_classes, depths=[ 2, 2, 5, 2 ], dims=96, ssm_d_state=1,ssm_ratio=2.0,ssm_rank_ratio=2.0,ssm_dt_rank="auto",ssm_act_layer="silu",ssm_conv=3,ssm_conv_bias=False,ssm_drop_rate=0.,ssm_init="v0",forward_type="v05_noz",mlp_ratio=4.0,mlp_act_layer="gelu",mlp_drop_rate=0.,drop_path_rate=0.2,patch_norm=True,norm_layer="ln2d",downsample_version="v3",patchembed_version="v2",gmlp=False,use_checkpoint=False,posembed=False,imgsize=224,)
    # if model_name == 'vmamba_s_x':
    #     model = vmamba_x(patch_size=4, in_chans=3, num_classes=num_classes, depths=[ 2, 2, 5, 2 ], dims=96, ssm_d_state=1,ssm_ratio=2.0,ssm_rank_ratio=2.0,ssm_dt_rank="auto",ssm_act_layer="silu",ssm_conv=3,ssm_conv_bias=False,ssm_drop_rate=0.,ssm_init="v0",forward_type="v05_noz",mlp_ratio=4.0,mlp_act_layer="gelu",mlp_drop_rate=0.,drop_path_rate=0.2,patch_norm=True,norm_layer="ln2d",downsample_version="v3",patchembed_version="v2",gmlp=False,use_checkpoint=False,posembed=False,imgsize=224,constraint_layers=constraint_layers)
    # if model_name == 'plain_mamba_l1':
    #     model = plain_mamba(arch='L1', drop_path_rate=0.2, num_classes=num_classes)
    # if model_name == 'plain_mamba_l1_x':
    #     model = plain_mamba_x(arch='L1', drop_path_rate=0.2, num_classes=num_classes, constraint_layers=constraint_layers)
    # if model_name == 'efficientvssm_tiny':
    #     model = efficientvssm(patch_size=4, in_chans=3, num_classes=num_classes, depths=[2, 2, 4, 2], dims=48, d_state=16,dt_rank="auto",ssm_ratio=2.0,attn_drop_rate=0.0,shared_ssm=False,softmax_version=False, drop_rate=0.0,drop_path_rate=0.2,mlp_ratio=0.,patch_norm=True, downsample_version="v1",use_checkpoint=False,window_size=2)
    # if model_name == 'efficientvssm_tiny_x':
    #     model = efficientvssm_x(patch_size=4, in_chans=3, num_classes=num_classes, depths=[2, 2, 4, 2], dims=48, d_state=16,dt_rank="auto",ssm_ratio=2.0,attn_drop_rate=0.0,shared_ssm=False,softmax_version=False, drop_rate=0.0,drop_path_rate=0.2,mlp_ratio=0.,patch_norm=True, downsample_version="v1",use_checkpoint=False,window_size=2, constraint_layers=constraint_layers)
    # if model_name == 'local_vim_tiny':
    #     model = local_vim_tiny(num_classes=num_classes, drop_path_rate=0.1)
    # if model_name == 'local_vim_tiny_x':
    #     model = local_vim_tiny_x(num_classes=num_classes, drop_path_rate=0.1, constraint_layers=constraint_layers)
    if model_name == 'localvim_tiny':
        model = local_vim_tiny(num_classes=num_classes)
    if model_name == 'localvim_tiny_x':
        model = local_vim_tiny_x(num_classes=num_classes, constraint_layers=constraint_layers)
    if model_name == 'simba_s':
        model = simba_s(num_classes=num_classes)
    if model_name == 'simba_s_x':
        model = simba_s_x(num_classes=num_classes, constraint_layers=constraint_layers)
    if model_name == 'vmamba_t':
        model = vmamba_t(num_classes=num_classes)
    if model_name == 'vmamba_t_x':
        model = vmamba_t_x(num_classes=num_classes, constraint_layers=constraint_layers)
    if model_name == 'efficientvmamba_t':
        model = efficientvmamba_t(num_classes=num_classes)
    if model_name == 'efficientvmamba_t_x':
        model = efficientvmamba_t_x(num_classes=num_classes, constraint_layers=constraint_layers)
    assert model is not None
    return model

# # for test
# def load_model(model_name, in_channels=3, num_classes=10, constraint_layers=None):
#     model = None
#     if model_name == 'vim_tiny':
#         model = vim_tiny(num_classes=num_classes)
#     if model_name == 'vim_tiny_x':
#         model = vim_tiny_x(num_classes=num_classes, constraint_layers=constraint_layers)
#
#     return model


def load_modules(model, model_layers=None):
    assert model_layers is None or type(model_layers) is list

    modules = []
    for module in model.modules():
        #     print(module)
        #     # if isinstance(module, torch.nn.Conv2d):
        #     #     modules.append(module)
        if isinstance(module, torch.nn.Linear):
            modules.append(module)
        # if isinstance(module, torch.nn.Identity):
        #     modules.append(module)
    # for name, module in model.named_modules():
    #     if 'attn.attn_drop' in name:
    #         modules.append(module)

    modules.reverse()  # reverse order
    if model_layers is None:
        model_modules = modules
    else:
        model_modules = []
        for layer in model_layers:
            model_modules.append(modules[layer])

    print('-' * 50)
    print('Model Layers:', model_layers)
    print('Model Modules:', model_modules)
    print('Model Modules Length:', len(model_modules))
    print('-' * 50)

    return model_modules


if __name__ == '__main__':
    from torchsummary import summary
    import sys

    sys.path.append('.')

    model = load_model('localvim_t')
    print(model)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    # summary(model, (3, 224, 224))
    print(y.shape)

    # modules = load_modules(model)
