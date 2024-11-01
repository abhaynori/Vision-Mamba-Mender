import os
import argparse
import shutil
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--num_epochs', default=300, type=int, help='num epochs')
    parser.add_argument('--model_dir', default='', type=str, help='model dir')
    parser.add_argument('--data_train_dir', default='', type=str, help='data dir')
    parser.add_argument('--data_test_dir', default='', type=str, help='data dir')
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

    print(args)
    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    # if os.path.exists(args.log_dir):
    #     shutil.rmtree(args.log_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL DIR:', args.model_dir)
    # print('LOG DIR:', args.log_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(args.model_name, num_classes=args.num_classes)
    # model.load_state_dict(torch.load('/nfs196/hjc/projects/Mamba/outputs//vim_tiny_imagenet50_bs256_lr0.001/models/model_ori.pth'))  # TODO
    model.to(device)

    print('*' * 20)
    model_ema = None
    if args.model_ema:  # false
        # if True:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,  # 0.99996
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print('model_ema', model_ema)

    train_loader = loaders.load_data(args.data_train_dir, args.data_name, data_type='train', batch_size=args.batch_size,
                                     args=args)
    test_loader = loaders.load_data(args.data_test_dir, args.data_name, data_type='test', batch_size=args.batch_size,
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
    # each epoch
    # ----------------------------------------
    since = time.time()

    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(args.num_epochs)):
        print('\n')
        loss, acc1, acc5 = train(train_loader, model, criterion, optimizer, device, mixup_fn, model_ema)
        writer.add_scalar(tag='training loss1', scalar_value=loss.avg, global_step=epoch)
        writer.add_scalar(tag='training acc1', scalar_value=acc1.avg, global_step=epoch)
        loss, acc1, acc5 = test(test_loader, model, device)
        writer.add_scalar(tag='test loss1', scalar_value=loss.avg, global_step=epoch)
        writer.add_scalar(tag='test acc1', scalar_value=acc1.avg, global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc < acc1.avg:
            best_acc = acc1.avg
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model_ori.pth'))
            if model_ema is not None:  # true
                torch.save(get_state_dict(model_ema), os.path.join(args.model_dir, 'model_ema.pth'))

        scheduler.step(epoch=epoch)

    print('COMPLETE !!!')
    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)
    print('TIME CONSUMED', time.time() - since)
    print('MODEL DIR', args.model_dir)


def train(train_loader, model, criterion, optimizer, device, mixup_fn, model_ema):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    dataload_time = AverageMeter('Data', ':6.3f')
    calculate_time = AverageMeter('Cal', ':6.3f')

    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss_meter, acc1_meter, acc5_meter, dataload_time, calculate_time])

    model.train()

    time_flag = time.time()

    # half_data_length = len(train_loader.dataset) // 2

    for i, samples in enumerate(train_loader):
        load_time = time.time() - time_flag

        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        if mixup_fn is not None:  # false
            inputs, labels = mixup_fn(inputs, labels)

        time_flag = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        cal_time = time.time() - time_flag

        if mixup_fn is not None:  # false
            labels = labels.argmax(dim=1)
        acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))

        loss_meter.update(loss.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))
        acc5_meter.update(acc5.item(), inputs.size(0))
        dataload_time.update(load_time, inputs.size(0))
        calculate_time.update(cal_time, inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        if model_ema is not None:
            model_ema.update(model)

        progress.display(i)
        time_flag = time.time()

        # if (i + 1) * len(inputs) >= half_data_length:
        #     break

    return loss_meter, acc1_meter, acc5_meter


def test(test_loader, model, device):
    # 使用默认的CELoss
    criterion = torch.nn.CrossEntropyLoss()

    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Test',
                             meters=[loss_meter, acc1_meter, acc5_meter])
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))

            loss_meter.update(loss.item(), inputs.size(0))
            acc1_meter.update(acc1.item(), inputs.size(0))
            acc5_meter.update(acc5.item(), inputs.size(0))

            progress.display(i)

    return loss_meter, acc1_meter, acc5_meter


if __name__ == '__main__':
    main()
