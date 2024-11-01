import os
import argparse
import shutil
import time
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from core.constraints import StateConstraint


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--data_train_dir', default='', type=str, help='data dir')
    parser.add_argument('--data_test_dir', default='', type=str, help='data dir')
    parser.add_argument('--num_epochs', default=200, type=int, help='num epochs')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--alpha', default=1e+7, type=float, help='weight of loss external')
    parser.add_argument('--beta', default=1e+7, type=float, help='weight of loss internal')
    parser.add_argument('--external_cache_layers', default=[], nargs='*', type=int, help='external cache layers')
    parser.add_argument('--internal_cache_layers', default=[], nargs='*', type=int, help='internal cache layers')
    parser.add_argument('--external_cache_types', default=[], nargs='*', type=str, help='external cache types')
    parser.add_argument('--internal_cache_types', default=[], nargs='*', type=str, help='internal cache types')
    parser.add_argument('--external_mask_dir', default=None, type=str, help='external mask dir')
    parser.add_argument('--internal_mask_dir', default=None, type=str, help='internal mask dir')
    parser.add_argument('--save_dir', default='', type=str, help='model dir')
    parser.add_argument('--log_dir', default='', type=str, help='log dir')

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='weight decay (default: 0.1)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # if os.path.exists(args.log_dir):
    #     shutil.rmtree(args.log_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL DIR:', args.model_path)
    print('ALPHA:', '{:e}'.format(args.alpha), '| BETA:', '{:e}'.format(args.beta))
    print('EXTERNAL CACHE LAYERS:', args.external_cache_layers, '| INTERNAL CACHE LAYERS:', args.internal_cache_layers)
    print('EXTERNAL CACHE TYPES:', args.external_cache_types, '| INTERNAL CACHE TYPES:', args.internal_cache_types)
    print('EXTERNAL MASK DIR:', args.external_mask_dir, '| INTERNAL MASK DIR:', args.internal_mask_dir)
    # print('LOG DIR:', args.log_dir)
    print('SAVE DIR:', args.save_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name,
                              num_classes=args.num_classes,
                              constraint_layers=args.external_cache_layers + args.internal_cache_layers)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    train_loader = loaders.load_data_mask(data_dir=args.data_train_dir,
                                          mask_dir=args.external_mask_dir,
                                          data_name=args.data_name,
                                          data_type='train',
                                          batch_size=args.batch_size,
                                          args=args)
    test_loader = loaders.load_data(data_dir=args.data_test_dir,
                                    data_name=args.data_name,
                                    data_type='test',
                                    batch_size=args.batch_size,
                                    args=args)

    mixup_fn = None
    mixup_active = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=args.smoothing, num_classes=args.num_classes)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:  # 0.1
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    scheduler, _ = create_scheduler(args, optimizer)

    writer = SummaryWriter(args.log_dir)

    # ----------------------------------------
    # optimizing configuration
    # ----------------------------------------
    constraint = StateConstraint(model=model,
                                 model_name=args.model_name,
                                 alpha=args.alpha,
                                 beta=args.beta,
                                 external_cache_layers=args.external_cache_layers,
                                 internal_cache_layers=args.internal_cache_layers,
                                 external_cache_types=args.external_cache_types,
                                 internal_cache_types=args.internal_cache_types,
                                 internal_mask_dir=args.internal_mask_dir)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(args.num_epochs)):
        print('\n')
        loss1, loss2, loss3, acc1 = train(train_loader, model, criterion, optimizer, constraint, device, mixup_fn)
        writer.add_scalar(tag='training loss1', scalar_value=loss1.avg, global_step=epoch)
        writer.add_scalar(tag='training loss2', scalar_value=loss2.avg, global_step=epoch)
        writer.add_scalar(tag='training loss3', scalar_value=loss3.avg, global_step=epoch)
        writer.add_scalar(tag='training acc1', scalar_value=acc1.avg, global_step=epoch)
        loss1, loss2, loss3, acc1 = test(test_loader, model, criterion, constraint, device)
        writer.add_scalar(tag='test loss1', scalar_value=loss1.avg, global_step=epoch)
        writer.add_scalar(tag='test loss2', scalar_value=loss2.avg, global_step=epoch)
        writer.add_scalar(tag='test loss3', scalar_value=loss3.avg, global_step=epoch)
        writer.add_scalar(tag='test acc1', scalar_value=acc1.avg, global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc < acc1.avg:
            best_acc = acc1.avg
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_ori.pth'))

        scheduler.step(epoch=epoch)

    print('COMPLETE !!! TIME CONSUMED:', time.time() - since)
    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)
    print('-' * 50)
    print('MODEL DIR:', args.model_path)
    print('ALPHA:', '{:e}'.format(args.alpha), '|| BETA:', '{:e}'.format(args.beta))
    print('EXTERNAL CACHE LAYERS:', args.external_cache_layers, '|| INTERNAL CACHE LAYERS:', args.internal_cache_layers)
    print('EXTERNAL CACHE TYPES:', args.external_cache_types, '|| INTERNAL CACHE TYPES:', args.internal_cache_types)
    print('EXTERNAL MASK DIR:', args.external_mask_dir, '|| INTERNAL MASK DIR:', args.internal_mask_dir)
    # print('LOG DIR:', args.log_dir)
    print('SAVE DIR:', args.save_dir)
    print('=' * 70)
    print('=' * 70)


def train(train_loader, model, criterion, optimizer, constraint, device, mixup_fn):
    loss1_meter = AverageMeter('Loss1', ':.4e')
    loss2_meter = AverageMeter('Loss2', ':.4e')
    loss3_meter = AverageMeter('Loss3', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss1_meter, loss2_meter, loss3_meter, acc1_meter])
    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels, masks = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)

        # # check inputs and masks
        # from utils import mask_util
        # fig_dir = os.path.join('/nfs196/hjc/projects/Mamba/outputs/test_utils')
        # mask_util.check_input_mask(inputs, masks, fig_dir, i)

        outputs = model(inputs)

        # loss CE
        loss1 = criterion(outputs, labels)

        if mixup_fn is not None:
            labels = labels.argmax(dim=1)

        # loss external
        loss2 = constraint.loss_external(outputs, labels, masks)

        # loss internal
        loss3 = constraint.loss_internal(outputs, labels)

        # total loss
        loss = loss1 + loss2 + loss3
        # loss = loss1 / (loss1.detach() + 1e-7) + loss2 / (loss2.detach() + 1e-7) + loss3 / (loss3.detach() + 1e-7)

        # acc
        acc1, = metrics.accuracy(outputs, labels)

        loss1_meter.update(loss1.item(), inputs.size(0))
        loss2_meter.update(loss2.item(), inputs.size(0))
        loss3_meter.update(loss3.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return loss1_meter, loss2_meter, loss3_meter, acc1_meter


def test(test_loader, model, criterion, constraint, device):
    criterion = torch.nn.CrossEntropyLoss()

    loss1_meter = AverageMeter('Loss1', ':.4e')
    loss2_meter = AverageMeter('Loss2', ':.4e')
    loss3_meter = AverageMeter('Loss3', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Test',
                             meters=[loss1_meter, loss2_meter, loss3_meter, acc1_meter])
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        # with torch.set_grad_enabled(False):
        outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        acc1, = metrics.accuracy(outputs, labels)
        loss2 = torch.zeros(loss1.shape)
        loss3 = torch.zeros(loss1.shape)

        loss1_meter.update(loss1.item(), inputs.size(0))
        loss2_meter.update(loss2.item(), inputs.size(0))
        loss3_meter.update(loss3.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))

        progress.display(i)

        # delete caches
        del inputs, outputs, loss1
        constraint.del_cache()
        # torch.cuda.empty_cache()

    return loss1_meter, loss2_meter, loss3_meter, acc1_meter


if __name__ == '__main__':
    main()
