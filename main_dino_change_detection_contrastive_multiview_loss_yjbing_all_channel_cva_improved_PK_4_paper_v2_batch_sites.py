
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import matplotlib.pyplot as plt
import copy
import utils_batch_oscd_one_img_train as utils
from torch.utils.data import DataLoader
from data.dataset_s2 import DFC2020
import metrics as metrics
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from vision_transformer import DINOHead
#使用vision transformer backbone时
import vision_transformer as vits
# #使用swin transformer backbone时
# import swin_transformer as vits

# # data augmentation
# from kornia import geometry as geo
# from kornia import filters
# from skimage import filters
# from util import adjust_learning_rate, AverageMeter, rosin, GaussianBlur, default, RandomApply

import sys
sys.path.append('E:/research_doctoral_laptop/my_code/')
import commonly_function as cf

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['SwinTransformer','vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--embed_dim', type=int, default=384,choices=[192,384,768,], help='swin transformer: 96; vit_small: 384 model tranfomer的embed_dim.')
    parser.add_argument('--pbatch_size', type=int, default=384,choices=[192,384,768,], help='swin transformer: 96; vit_small: 384 batch_size of patches during inference.')

    parser.add_argument('--out_dim', default=6000, choices=[600,65536],type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases. 
        这个值越大loss越小，这个值越小loss越大
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--sample_number',default=2,type=int,help = 'Number of dataset.')
    parser.add_argument('--eps',default = 1e-7,type=float,help = 'constant in calculation loss.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # adjust
    # input data chans
    parser.add_argument('--use_s2hr', action='store_true', default=False, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--all_chans', action='store_true', default=True, help='whether using all chans')
    parser.add_argument('--use_s1', action='store_true', default=False, help='use mean feature map, input img chanels add one')
    parser.add_argument('--use_s2', action='store_true', default=False, help='use mbi cao feature map, input img chanels add one')
    parser.add_argument('--use_s3', action='store_true', default=False, help='use harris feature map, input img chanels add one')
    parser.add_argument('--use_s4', action='store_true', default=False, help='use variance feature map, input img chanels add one')
    # input data
    parser.add_argument('--in_chans', type=int, default=12, help='读取图像的通道数')
    parser.add_argument('--calculate_chans', type=int, default=4, help='输入网络计算的通道数, 也表示网络模型可输入的通道')
    parser.add_argument('--input_img_chans', type=int, default=4, help='读取图像用于网络计算的图像通道数')
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')

    # parser.add_argument('--data_dir_train', type=str, default='E:/research_doctoral_laptop/data/change_detection/chenyuxing', help='path to training data set')
    # parser.add_argument('--data_dir_eval', type=str, default='E:/research_doctoral_laptop/data/change_detection/chenyuxing', help='path to test data set')

    parser.add_argument('--data_dir_train', type=str, default='E:/research_doctoral_laptop/data/change_detection/OSCD/train_1_image/chongqing', help='path to training data set')
    parser.add_argument('--data_dir_eval', type=str, default='E:/research_doctoral_laptop/data/change_detection/OSCD/train_1_image/chongqing', help='path to test data set') #'E:/research_doctoral_laptop/data/change_detection/OSCD/train_1_image/montpellier' 'E:/research_doctoral_laptop/data/change_detection/OSCD/GaofeBefore'
    parser.add_argument('--oscd_data_name', type=str, default='chongqing', help='oscd 数据的名称')

    # parser.add_argument('--data_dir_train', type=str, default='E:/research_doctoral_laptop/data/change_detection/OSCD/GaofeAfter2', help='path to training data set')
    # parser.add_argument('--data_dir_eval', type=str, default='E:/research_doctoral_laptop/data/change_detection/OSCD/GaofeAfter2', help='path to test data set') #'E:/research_doctoral_laptop/data/change_detection/OSCD/train_1_image/montpellier' 'E:/research_doctoral_laptop/data/change_detection/OSCD/GaofeBefore'
    # parser.add_argument('--oscd_data_name', type=str, default='GaofeAfter2', help='oscd 数据的名称')

    #output
    parser.add_argument('--output_dir', default="./chongqing_test", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--eval_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--binary_output', default='chenyuxing', type=str,choices=['chenyuxing', 'otsu'],
                        help='method to generate change map. chenyuxing: binary map, otsu: binary map', )
    parser.add_argument('--multicalss_output', default='no', choices=['class_change', 'class'],type=str,
                        help='class_change:对变化区域分类, class:对直接的预测图进行分类, no:不执行分类')
    parser.add_argument('--n_clusters', default=7, type=int,
                        help='the number of clusters using K Means method,')

    #train
    parser.add_argument('--cca', default=0.5, type=float, help='the cca parameter of hard loss')
    parser.add_argument('--att_type', default='NAM', type=str, choices=['BAM', 'CBAM','NAM'],help='使用注意力机制')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels
            of input square patches - default 16 (for 16x16 patches). Using smaller
            values leads to better performance but requires more memory. Applies only
            for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
            mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size for data training')
    parser.add_argument('--crop_size', type=int, default=200, help='crop_size for ensuring same patch_size within all batches')
    parser.add_argument('--unfold_stride', type=int, default=4, help='stride during the training patches')
    parser.add_argument('--val_unfold_stride', type=int, default=1, help='stride during the inference patches')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--loss_function', default='HardNegtive', type=str, choices=['dino', 'NCECriterion','Similarity','BarlowTwins','HardNegtive'], help='flag for testing on test data set')
    parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',help='BarlowTwins loss function weight on off-diagonal terms')
    parser.add_argument('--resume', default=False, type=bool, help='flag for training from checkpoint')
    parser.add_argument('--test', default=False, type=bool, help='flag for testing on test data set')

    return parser

def get_train_val_loader(args):
    data_set = DFC2020(args.data_dir_train,
                    subset="train",
                    no_savanna=args.no_savanna,
                    use_s2hr=args.use_s2hr,
                    use_s2mr=args.use_s2mr,
                    use_s2lr=args.use_s2lr,
                    use_s1=args.use_s1,
                    use_s2=args.use_s2,
                    use_s3=args.use_s3,
                    use_s4=args.use_s4,
                    unlabeled=True,
                    transform=True,#设置为False表示，不裁剪，batch_size 必须设置为1，应为每个数据的大小不同
                    train_index=None,
                    crop_size=args.crop_size,
                    in_chans=args.in_chans,
                    all_chans=args.all_chans)
    n_classes = data_set.n_classes
    n_inputs = data_set.n_inputs

    eval_set = DFC2020(args.data_dir_eval,
                       subset="train",
                       no_savanna=args.no_savanna,
                       use_s2hr=args.use_s2hr,
                       use_s2mr=args.use_s2mr,
                       use_s2lr=args.use_s2lr,
                       use_s1=args.use_s1,
                       use_s2=args.use_s2,
                       use_s3=args.use_s3,
                       use_s4=args.use_s4,
                       unlabeled=False, #unlabeled 是否加载label, False:表示加载label
                       transform=False, #这里的transform 是否随机裁剪
                       train_index=None,
                       crop_size=args.crop_size,
                       in_chans=args.in_chans,
                       all_chans=args.all_chans)

    # set up dataloaders
    train_loader = DataLoader(data_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)#True表示如果最后剩下不完全的batch,丢弃。False表示不丢弃。

    eval_loader = DataLoader(eval_set,
                              batch_size=1,#args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader, eval_loader, n_inputs, n_classes

def NCE_loss(x,n_data,eps):   #x.shape (2,16385,1)
    #归一化)
    for i in range(0,x.shape[0]):
        x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i])-torch.min(x[i]))
    bsz = x.shape[0]
    m = x.size(1) - 1

    # noise distribution
    Pn = 1 / float(n_data)

    # loss for positive pair
    P_pos = x.select(1, 0) #每张图取一个正样本
    #torch.div(a,b):a/b   .log_(a):以e为底数求a的对数  a.add(b):对a所有元素+b
    log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

    # loss for K negative pair
    P_neg = x.narrow(1, 1, m) #每张图取前一个正样本以外的为负样本
    log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + self.eps)).log_()
    loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

    return loss

def train_dino(args,i):
    utils.init_distributed_mode(args,i)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_loader, eval_loader, n_inputs, n_classes = get_train_val_loader(args)
    args.n_inputs = n_inputs
    args.n_classes = 2

    print(f"Train data loaded: there are {len(train_loader.sampler)} images.")
    print(f"Val data loaded: there are {len(eval_loader.sampler)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            in_chans = args.calculate_chans,
            embed_dim = args.embed_dim,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size,
                                           in_chans = args.calculate_chans,
                                           embed_dim = args.embed_dim)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions, projection head
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    # define loss function
    if args.loss_function == 'dino':
        #dino loss
        dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).cuda()
        #NCE loss, contrastive multiview loss
    elif args.loss_function == 'NCECriterion':
        dino_loss = NCECriterion_loss(
            args.sample_number,
            args.eps,
            args.local_crops_number + 2,
            args.out_dim,
            args.teacher_temp,
            args.warmup_teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs).cuda()
    elif args.loss_function == 'HardNegtive':
        dino_loss = HardNegtive_loss(
            args.sample_number,
            args.eps,
            args.local_crops_number + 2,
            args.out_dim,
            args.teacher_temp,
            args.warmup_teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs).cuda()
    elif args.loss_function == 'Similarity':
        dino_loss = Similarity(
            args.sample_number,
            args.eps,
            args.local_crops_number + 2,
            args.out_dim,
            args.teacher_temp,
            args.warmup_teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs).cuda()
    elif args.loss_function == 'BarlowTwins':
        dino_loss = BarlowTwins_loss(args).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups,lr=0.0005)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    # lr_schedule = np.arange(args.lr,args.min_lr,-(args.lr-args.min_lr)/args.epochs)
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(train_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ...  ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    max_miou = 0
    for epoch in range(start_epoch, args.epochs):
        # before_data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... 开始训练 ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            train_loader,eval_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # evaluate the model

        if (epoch + 1) % args.eval_freq == 0:
            for idx, (batch, _) in enumerate(eval_loader):
                # miou,img_name,aa ,cva_miou,cva_aa = validate(batch,teacher,epoch+1)  ###这里的student 确定先验知识融合方法后，需要换成teacher
                miou,img_name,aa = validate(batch,teacher,epoch+1)  ###这里的student 确定先验知识融合方法后，需要换成teacher
                teacher.train()  ###这里的student 确定先验知识融合方法后，需要换成teacher
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,'MIoU': miou,'aa': aa,'image name': img_name}
                # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                #              'epoch': epoch,'MIoU': miou,'aa': aa,'cva MIoU': cva_miou,'cva aa': cva_aa,'image name': img_name}
                if utils.is_main_process():
                    with (Path(args.output_dir) / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

        # save checkpoints # ============ writing logs ... 保持该epoch训练模型============
        if (epoch + 1) % args.eval_freq == 0:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            # if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            if miou > max_miou:
                max_miou = miou
                utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def compute_squared_l2_distance(pred: torch.Tensor, surrogate_label: torch.Tensor) -> torch.Tensor:

    losses = (pred - surrogate_label) ** 2
    losses = losses.view(losses.shape[0], -1)
    losses = torch.mean(losses, dim=1)
    losses = losses.cpu().detach()

    return losses

def compute_heatmap(img: torch.Tensor, patch_size, unfold_stride,student):

    """
    img.shape -> (B, C, iH, iW)
    B  : batch size
    C  : channels of image (same to patches.shape[1])
    iH : height of image
    iW : width of image

    patches.shape -> (P, C, pH, pW)
    P  : patch size
    C  : channels of image (same to img.shape[1])
    pH : height of patch
    pW : width of patch
    """

    patches = patchize(img, patch_size, unfold_stride)

    B, C, iH, iW = img.shape
    P, C, pH, pW = patches.shape

    heatmap = torch.zeros(P)
    heatmap_patch1 = torch.zeros(P)
    heatmap_patch2 = torch.zeros(P)
    quotient, remainder = divmod(P, args.pbatch_size)

    for i in range(quotient):

        start = i * args.pbatch_size
        end = start + args.pbatch_size

        patch = patches[start:end, :, :, :]
        patch = patch.to('cuda')

        patch1, patch2 = torch.split(patch, [C//2, C//2], dim=1)
        surrogate_label = student(patch1)
        pred = student(patch2)

        patch1_sub = torch.mean(surrogate_label, dim=1)
        patch1_sub = patch1_sub.cpu().detach()
        heatmap_patch1[start:end]= patch1_sub
        patch2_sub =torch.mean(pred, dim=1)
        patch2_sub = patch2_sub.cpu().detach()
        heatmap_patch2[start:end]= patch2_sub

        losses = compute_squared_l2_distance(pred, surrogate_label)
        heatmap[start:end] = losses

    if remainder != 0:
        patch = patches[-remainder:, :, :, :]
        patch = patch.to('cuda')
        patch1, patch2 = torch.split(patch, [C//2, C//2], dim=1)
        surrogate_label = student(patch1)
        pred = student(patch2)

        patch1_sub = torch.mean(surrogate_label, dim=1)
        patch1_sub = patch1_sub.cpu().detach()
        heatmap_patch1[-remainder:]= patch1_sub
        patch2_sub =torch.mean(pred, dim=1)
        patch2_sub = patch2_sub.cpu().detach()
        heatmap_patch2[-remainder:]= patch2_sub

        losses = compute_squared_l2_distance(pred, surrogate_label)
        heatmap[-remainder:] = losses

    fold = nn.Fold(
        output_size=(iH, iW),
        kernel_size=(pH, pW),
        stride=unfold_stride,
    )

    heatmap_patch1 = heatmap_patch1.expand(B, pH * pW, P)#[1, 64, 129472]
    heatmap_patch1 = fold(heatmap_patch1)#[1, 1, 903, 585]
    heatmap_patch1 = heatmap_patch1.squeeze()#[903, 585]
    heatmap_patch2 = heatmap_patch2.expand(B, pH * pW, P)#[1, 64, 129472]
    heatmap_patch2 = fold(heatmap_patch2)#[1, 1, 903, 585]
    heatmap_patch2 = heatmap_patch2.squeeze()#[903, 585]

    heatmap = heatmap.expand(B, pH * pW, P)#[1, 64, 129472]
    heatmap = fold(heatmap)#[1, 1, 903, 585]
    heatmap = heatmap.squeeze()#[903, 585]

    del patches
    return heatmap,heatmap_patch1,heatmap_patch2

def multiclass_k_means(diff_img,binary_change_map, n_clusters,):

    # Calculate the weighting feature direction θ: theta shape (650,650)
    # theta = np.arccos(1 / np.sqrt(FILTER_NUM[-1]) * (np.sum(eig_value * diff_img, -1) / np.sqrt(
    #     np.sum(np.square(eig_value)) * np.sum(np.square(diff_img), axis=-1))))
    theta = np.arccos(1 / np.sqrt(4) * (np.sum(np.reshape(diff_img.numpy(),(diff_img.numpy().shape[0],diff_img.numpy().shape[1],1)), -1) / np.sqrt(
        np.sum(np.square(np.reshape(diff_img.numpy(),(diff_img.numpy().shape[0],diff_img.numpy().shape[1],1)))))))
    print('-------Perform Multi-class Change Detection-------')
    # binary_change_map = binary_change_map.numpy()*255
    binary_change_map_2 = copy.deepcopy(binary_change_map)
    binary_change_map_2[binary_change_map_2==False]=True
    changed_idx = binary_change_map_2
    # #不使用投影为the weighting feature direction θ
    # changed_theta = diff_img[changed_idx]
    changed_theta = theta[changed_idx]
    changed_theta = np.reshape(changed_theta, (-1, 1))
    #使用K均值分类多类变化图
    KMeans_cls = KMeans(n_clusters=n_clusters, max_iter=1500)
    print('-------Clustering algorithm is running')
    KMeans_cls.fit(changed_theta)
    label_pred = KMeans_cls.labels_  # get cluster label
    height, width = binary_change_map_2.shape
    multi_change_map = np.zeros((height, width, 3))
    k = 0
    for h in range(height):
        for w in range(width):
            if changed_idx[h, w]:
                if label_pred[k] == 0:
                    multi_change_map[h, w] = np.array([255, 255, 0])
                elif label_pred[k] == 1:
                    multi_change_map[h, w] = np.array([255, 0, 0])
                elif label_pred[k] == 2:
                    multi_change_map[h, w] = np.array([0, 0, 255])
                k += 1
    return torch.from_numpy(multi_change_map)

def multiclass_change_map_k_means(diff_img,binary_change_map, n_clusters,):

    # Calculate the weighting feature direction θ: theta shape (650,650)
    # theta = np.arccos(1 / np.sqrt(FILTER_NUM[-1]) * (np.sum(eig_value * diff_img, -1) / np.sqrt(
    #     np.sum(np.square(eig_value)) * np.sum(np.square(diff_img), axis=-1))))
    theta = np.arccos(1 / np.sqrt(4) * (np.sum(np.reshape(diff_img.numpy(),(diff_img.numpy().shape[0],diff_img.numpy().shape[1],1)), -1) / np.sqrt(
        np.sum(np.square(np.reshape(diff_img.numpy(),(diff_img.numpy().shape[0],diff_img.numpy().shape[1],1)))))))
    print('-------Perform Multi-class Change Detection-------')
    # binary_change_map = binary_change_map.numpy()*255
    # binary_change_map[binary_change_map==False]=True
    changed_idx = binary_change_map
    # #不使用投影为the weighting feature direction θ
    # changed_theta = diff_img[changed_idx]
    changed_theta = theta[changed_idx]
    changed_theta = np.reshape(changed_theta, (-1, 1))
    #使用K均值分类多类变化图
    KMeans_cls = KMeans(n_clusters=n_clusters, max_iter=1500)
    print('-------Clustering algorithm is running')
    KMeans_cls.fit(changed_theta)
    label_pred = KMeans_cls.labels_  # get cluster label
    height, width = binary_change_map.shape
    multi_change_map = np.zeros((height, width, 3))
    k = 0
    for h in range(height):
        for w in range(width):
            if changed_idx[h, w]:
                if label_pred[k] == 0:
                    multi_change_map[h, w] = np.array([255, 255, 0])
                elif label_pred[k] == 1:
                    multi_change_map[h, w] = np.array([255, 0, 0])
                elif label_pred[k] == 2:
                    multi_change_map[h, w] = np.array([0, 0, 255])
                k += 1
    return torch.from_numpy(multi_change_map)

def change_map_otsu(difference_img):
    threshold, cd_map = cf.otsu(difference_img.numpy())
    return torch.from_numpy(cd_map >= 255)

def change_map(difference_img):

    threshold_init = torch.mean(difference_img) + 3 * torch.std(difference_img)
    difference_img = torch.where(difference_img < threshold_init, difference_img, threshold_init) #若满足条件取，difference_img的值，否者取threshold_init值
    #tmp = difference_img.cpu().detach().numpy()
    #threshold = filters.threshold_otsu(difference_img)
    #rosin methods
    difference_img = (difference_img - difference_img.mean()) / difference_img.std()
    #threshold = rosin(difference_img)
    threshold = difference_img.min().abs()

    return difference_img >= threshold

def validate(batch,student,epochs):

    # switch to evaluate mode
    student.eval()

    # main validation loop
    conf_mat = metrics.ConfMatrix(args.n_classes, 100)
    cva_conf_mat = metrics.ConfMatrix(args.n_classes, 100)

    with torch.no_grad():
        # for idx, (batch, _) in enumerate(val_loader):

        # unpack sample
        image, target = batch['image'], batch['label']
        image_name = batch['id'][0].split('_')[0]
        start = time.time()
        # ===================forward=====================
        image_1, image_2 = torch.split(image, [args.in_chans, args.in_chans], dim=1)

        image = torch.cat([image_1[:,0:args.calculate_chans,:,:],image_2[:,0:args.calculate_chans,:,:]],dim=1)
        prediction,prediction_patch1,prediction_patch2 = compute_heatmap(image, args.patch_size, args.val_unfold_stride,student)

        #####特征图计算###########
        batch_view_1_ndvi = torch.unsqueeze((image_1[:,7,:,:]-image_1[:,3,:,:])/(image_1[:,7,:,:]+image_1[:,3,:,:]),dim=1)
        batch_view_2_ndvi = torch.unsqueeze((image_2[:,7,:,:]-image_2[:,3,:,:])/(image_2[:,7,:,:]+image_2[:,3,:,:]),dim=1)
        batch_view_1_ndwi = torch.unsqueeze((image_1[:,2,:,:]-image_1[:,7,:,:])/(image_1[:,2,:,:]+image_1[:,7,:,:]),dim=1)
        batch_view_2_ndwi = torch.unsqueeze((image_2[:,2,:,:]-image_2[:,7,:,:])/(image_2[:,2,:,:]+image_2[:,7,:,:]),dim=1)
        batch_view_1_ndbsi = torch.unsqueeze((image_1[:,3,:,:]-image_1[:,1,:,:])/(image_1[:,3,:,:]+image_1[:,1,:,:]),dim=1)
        batch_view_2_ndbsi = torch.unsqueeze((image_2[:,3,:,:]-image_2[:,1,:,:])/(image_2[:,3,:,:]+image_2[:,1,:,:]),dim=1)
        # batch_view_1_ndbi = torch.unsqueeze((image_1[:,7,:,:]-image_1[:,11,:,:])/(image_1[:,7,:,:]+image_1[:,11,:,:]),dim=1)
        # batch_view_2_ndbi = torch.unsqueeze((image_2[:,7,:,:]-image_2[:,11,:,:])/(image_2[:,7,:,:]+image_2[:,11,:,:]),dim=1)

        batch_view_1_sei = torch.unsqueeze(((image_1[:,0,:,:]+image_1[:,8,:,:])-(image_1[:,2,:,:]+image_1[:,7,:,:]))/((image_1[:,0,:,:]+image_1[:,8,:,:])+(image_1[:,2,:,:]+image_1[:,7,:,:])),dim=1)
        batch_view_2_sei= torch.unsqueeze(((image_2[:,0,:,:]+image_2[:,8,:,:])-(image_2[:,2,:,:]+image_2[:,7,:,:]))/((image_2[:,0,:,:]+image_2[:,8,:,:])+(image_2[:,2,:,:]+image_2[:,7,:,:])),dim=1)


        # batch_view_1_nir = torch.unsqueeze((batch_view_1[:,7,:,:]),dim=1)
        # batch_view_2_nir = torch.unsqueeze((batch_view_2[:,7,:,:]),dim=1)

        batch_view_1_features = torch.cat([batch_view_1_ndvi,batch_view_1_ndwi,batch_view_1_ndbsi,batch_view_1_sei],dim=1)
        batch_view_2_features = torch.cat([batch_view_2_ndvi,batch_view_2_ndwi,batch_view_2_ndbsi,batch_view_1_sei],dim=1)

        image_features = torch.cat([batch_view_1_features,batch_view_2_features],dim=1)
        # student_features = copy.deepcopy(student) #增加网络输入通道，用于增加特征，但无法计算
        # student_features.backbone.patch_embed.proj = nn.Conv2d(batch_view_1_features.shape[1], args.embed_dim, kernel_size=student.backbone.patch_embed.proj.kernel_size, stride=student.backbone.patch_embed.proj.stride) #增加网络输入通道，用于增加特征，但无法计算
        prediction_features,prediction_patch1,prediction_patch2 = compute_heatmap(image_features, args.patch_size, args.val_unfold_stride,student)
        prediction = prediction + 0.5*prediction_features

        # image_1_imgfeatures = torch.cat([image_1[:,0:args.input_img_chans,:,:],batch_view_1_features],dim=1)
        # image_2_imgfeatures = torch.cat([image_2[:,0:args.input_img_chans,:,:],batch_view_2_features],dim=1)
        # image  = torch.cat([image_1_imgfeatures,image_2_imgfeatures],dim=1)
        # prediction,prediction_patch1,prediction_patch2 = compute_heatmap(image, args.patch_size, args.val_unfold_stride,student)

        #####特征图计算###########

        print('time elapsed:', time.time() - start)

        #####CVA
        # def feature (img1,img2):
        #     a_feature_map = np.load(img1)['data']
        #     b_feature_map = np.load(img2)['data']
        #     feature_diff = (a_feature_map-b_feature_map)**2
        #     feature_diff = (feature_diff-np.mean(feature_diff))/np.std(feature_diff)
        #     return  feature_diff
        #
        # path = args.data_dir_train+'/ROIs0000_autumn/'
        # # path = 'E:/research_doctoral_laptop/data/change_detection/OSCD/train_1_image/'+args.oscd_data_name+'/ROIs0000_autumn/'
        # data_name = args.oscd_data_name
        # ndwi_img1 = path + 'as2_0/'+data_name+'_as2_ndwi.npz'
        # ndwi_img2 = path + 'bs2_0/'+data_name+'_bs2_ndwi.npz'
        # ndwi_feature_diff = feature(ndwi_img1,ndwi_img2)
        # _, ndwi_cd_map = cf.otsu(ndwi_feature_diff)
        # ndwi_cd_number_sum = np.sum(ndwi_cd_map == 255)
        #
        # ndvi_img1 = path + 'as2_0/'+data_name+'_as2_ndvi.npz'
        # ndvi_img2 = path + 'bs2_0/'+data_name+'_bs2_ndvi.npz'
        # ndvi_feature_diff = feature(ndvi_img1,ndvi_img2)
        # _, ndvi_cd_map = cf.otsu(ndvi_feature_diff)
        # ndvi_cd_number_sum = np.sum(ndvi_cd_map == 255)
        #
        # # mean_img1 = path + 'as2_0/'+data_name+'_as2_mean.npz'
        # # mean_img2 = path + 'bs2_0/'+data_name+'_bs2_mean.npz'
        # # mean_feature_diff = feature(mean_img1,mean_img2)
        #
        # sei_img1 = path + 'as2_0/'+data_name+'_as2_sei.npz'
        # sei_img2 = path + 'bs2_0/'+data_name+'_bs2_sei.npz'
        # sei_feature_diff = feature(sei_img1,sei_img2)
        # _, sei_cd_map = cf.otsu(sei_feature_diff)
        # sei_cd_number_sum = np.sum(sei_cd_map == 255)

        # bpi_img1 = path + 'as2_0/'+data_name+'_as2_bpi.npz'
        # bpi_img2 = path + 'bs2_0/'+data_name+'_bs2_bpi.npz'
        # bpi_feature_diff = feature(bpi_img1,bpi_img2)
        # _, bpi_cd_map = cf.otsu(bpi_feature_diff)
        # bpi_cd_number_sum = np.sum(bpi_cd_map == 255)

        # band9_11_img1 = path + 'as2_0/'+data_name+'_as2_band9_11.npz'
        # band9_11_img2 = path + 'bs2_0/'+data_name+'_bs2_band9_11.npz'
        # band9_11_feature_diff = feature(band9_11_img1,band9_11_img2)

        # cva_prediction = 0.8*(prediction-torch.mean(prediction))/torch.std(prediction)+0.1*ndvi_feature_diff+0.05*ndwi_feature_diff+0.03*sei_feature_diff+0.02*bpi_feature_diff#+0.02*band9_11_feature_diff
        #####CVA

        #生成二值变化图
        if args.binary_output == 'chenyuxing':
            cd_map = change_map(prediction)
            cd_number_sum = torch.sum(cd_map == True)
            # cva_cd_map = change_map(cva_prediction)
        elif args.binary_output == 'otsu':
            cd_map = change_map_otsu(prediction)
            cd_number_sum = torch.sum(cd_map == True)
            # cva_cd_map = change_map_otsu(cva_prediction)

        # a2,a3,a4,a5 = ndvi_cd_number_sum/cd_number_sum,ndwi_cd_number_sum/cd_number_sum,sei_cd_number_sum/cd_number_sum,bpi_cd_number_sum/cd_number_sum

        #v1：精度有所提升
        # a2,a3,a4 = ndvi_cd_number_sum/(cd_map.shape[0]*cd_map.shape[1])/10,ndwi_cd_number_sum/(cd_map.shape[0]*cd_map.shape[1])/10,sei_cd_number_sum/(cd_map.shape[0]*cd_map.shape[1])#,bpi_cd_number_sum/(cd_map.shape[0]*cd_map.shape[1])
        # a1 = 1-a2-a3

        #v2：精度反而下降了
        # all_features_change_number = cd_number_sum+ndvi_cd_number_sum+ndwi_cd_number_sum
        # a2,a3,a4 = ndvi_cd_number_sum/all_features_change_number,ndwi_cd_number_sum/all_features_change_number,sei_cd_number_sum/all_features_change_number#,bpi_cd_number_sum/(cd_map.shape[0]*cd_map.shape[1])
        # a1 = cd_number_sum/all_features_change_number
        # a1 = 1-a2-a3#-a4

        #v3：精度反而下降了
        # a1 = cd_number_sum / (cd_number_sum+ndvi_cd_number_sum+ndwi_cd_number_sum)
        # a2 = ndvi_cd_number_sum / (cd_number_sum+ndvi_cd_number_sum+ndwi_cd_number_sum)/10
        # a3 = ndwi_cd_number_sum / (cd_number_sum+ndvi_cd_number_sum+ndwi_cd_number_sum)/10
        # # a4 = sei_cd_number_sum / (cd_number_sum+ndvi_cd_number_sum+ndwi_cd_number_sum+sei_cd_number_sum)
        # a1 = 1-a2-a3

        #v4：精度
        # prediction_normalize = (prediction-torch.mean(prediction))/torch.std(prediction)
        # # ndvi_feature_diff = (ndvi_feature_diff-np.min(ndvi_feature_diff))/(np.max(ndvi_feature_diff)-np.min(ndvi_feature_diff))
        # # ndwi_feature_diff = (ndwi_feature_diff-np.min(ndwi_feature_diff))/(np.max(ndwi_feature_diff)-np.min(ndwi_feature_diff))
        # r1 = 1/torch.sqrt(torch.sum((prediction_normalize.flatten() - ndvi_feature_diff.flatten()) ** 2) / len(ndvi_feature_diff.flatten()))
        # r2 = 1/torch.sqrt(torch.sum((prediction_normalize.flatten() - ndwi_feature_diff.flatten()) ** 2) / len(ndwi_feature_diff.flatten()))
        # r3 = 1/np.sqrt(np.sum((ndvi_feature_diff.flatten() - ndwi_feature_diff.flatten()) ** 2) / len(ndwi_feature_diff.flatten()))
        # a2 = (r3/2)/(r1+r2+r3)
        # a3 = (r3/2)/(r1+r2+r3)
        # a1 = (r1+r2)/(r1+r2+r3)

        # print('a1: %4f, a2: %4f, a3: %4f'%(a1,a2,a3))#,a5))
        # print('a1: %4f, a2: %4f, a3: %4f, a4: %4f'%(a1,a2,a3,a4))#,a5))
        # print('a1: %4f, a2: %4f, a3: %4f, a4: %4f, a5: %4f'%(a1,a2,a3,a4,a5))
        # cva_prediction = a1*(prediction-torch.mean(prediction))/torch.std(prediction)+a2*ndvi_feature_diff+a3*ndwi_feature_diff#+a4*sei_feature_diff#+a5*bpi_feature_diff
        # cva_prediction = a1*(prediction-torch.mean(prediction))/torch.std(prediction)+a2*ndvi_feature_diff+a3*ndwi_feature_diff#+a4*sei_feature_diff#+a5*bpi_feature_diff

        # if args.binary_output == 'chenyuxing':
        #     cva_cd_map = change_map(cva_prediction)
        # elif args.binary_output == 'otsu':
        #     cva_cd_map = change_map_otsu(cva_prediction)

        # if args.multicalss_output == 'class_change':
        #     multiclass_cd_map = multiclass_change_map_k_means(prediction,cd_map,args.n_clusters)
        # elif args.multicalss_output == 'class':
        #     multiclass_cd_map = multiclass_k_means(prediction,cd_map,args.n_clusters)

        if args.multicalss_output == 'class_change':
            multiclass_cd_map = multiclass_change_map_k_means(prediction,cd_map,args.n_clusters)
            import imageio
            imageio.imwrite(args.output_dir + '/'+image_name+'_Multiclass_CD_val_'+str(epochs)+'.png', multiclass_cd_map.numpy().astype(np.uint8))
        elif args.multicalss_output == 'class':
            multiclass_cd_map = multiclass_k_means(prediction,cd_map,args.n_clusters)
            import imageio
            imageio.imwrite(args.output_dir + '/'+image_name+'_Multiclass_CD_val_'+str(epochs)+'.png', multiclass_cd_map.numpy().astype(np.uint8))


        plt.imsave(args.output_dir+'/'+image_name+'_CDI_val_'+str(epochs)+'.png', prediction.squeeze().cpu().detach().numpy(), cmap='gray')
        plt.imsave(args.output_dir+'/'+image_name+'_patch1_CDI_val_'+str(epochs)+'.png', prediction_patch1.squeeze().cpu().detach().numpy())
        plt.imsave(args.output_dir+'/'+image_name+'_patch2_CDI_val_'+str(epochs)+'.png', prediction_patch2.squeeze().cpu().detach().numpy())
        plt.imsave(args.output_dir+'/'+image_name+'_CD_val_'+str(epochs)+'.png', np.squeeze(cd_map), cmap='gray')
        # plt.imsave(args.output_dir+'/'+image_name+'_CD_cva_val_'+str(epochs)+'.png', np.squeeze(cva_cd_map), cmap='gray')
        # plt.imsave(args.output_dir+'/Multiclass_CD_val_'+str(epochs)+'.png', np.squeeze(multiclass_cd_map))
        # import imageio
        # imageio.imwrite(args.output_dir + '/'+image_name+'_Multiclass_CD_val_'+str(epochs)+'.png', multiclass_cd_map.numpy().astype(np.uint8))
        # calculate error metrics
        if 255 in target.cpu().numpy():
            target[target==255]=1
        conf_mat.add_batch(target.cpu().numpy(), np.expand_dims(cd_map, axis=0))
        # cva_conf_mat.add_batch(target.cpu().numpy(), np.expand_dims(cva_cd_map, axis=0))

        # close progressbar
        print("[Val] MIoU: {:.2f}%".format(conf_mat.get_mIoU() * 100))
        print("[Val] aa: {:.2f}%".format(conf_mat.get_aa() * 100))
        # print("[cva Val] MIoU: {:.2f}%".format(cva_conf_mat.get_mIoU() * 100))
        # print("[cva Val] aa: {:.2f}%".format(cva_conf_mat.get_aa() * 100))
        return conf_mat.get_mIoU() * 100, image_name, conf_mat.get_aa() * 100 #, cva_conf_mat.get_mIoU() * 100, cva_conf_mat.get_aa() * 100

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, train_loader,eval_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    # after_img_list=[]
    # for _, (batch, _) in enumerate(train_loader):
    #     after_img_list.append(after_images)   #这个地方有个bug需检查，after_data_loader和before_data_loader对应的data顺序是否就是相同
    for it, (batch, _) in enumerate(train_loader):
        # after_images = after_img_list[it]
        image = batch['image'].cuda(non_blocking=True)
        # update weight decay and learning rate according to their schedule
        it = len(train_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        patches = patchize(image, args.patch_size, args.unfold_stride)
        P, C, pH, pW = patches.shape
        # random shuffle index
        shuffle_ids = torch.randperm(P).cuda()
        # shuffle for training
        this_patches = patches[shuffle_ids]
        # training in each pbatch_size
        quotient, remainder = divmod(P, args.pbatch_size)
        pbatch = quotient if quotient > 0 else remainder

        # move images to gpu
        # before_images = [im.cuda(non_blocking=True) for im in before_images]
        # after_images = [im.cuda(non_blocking=True) for im in after_images]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # teacher_output = teacher(before_images[:2])  # only the 2 global views pass through the teacher
            # #输入图像data
            # student_output = student(before_images)
            # before_loss = dino_loss(student_output, teacher_output, epoch)
            train_loss = 0.0
            for i in range(pbatch):
                start = i * args.pbatch_size
                end = start + args.pbatch_size
                patch = this_patches[start:end, :, :, :]
                batch_view_1, batch_view_2 = torch.split(patch, [args.in_chans, args.in_chans], dim=1)
                batch_view_1_1234 = batch_view_1[:,0:args.input_img_chans,:,:]
                batch_view_2_1234 = batch_view_2[:,0:args.input_img_chans,:,:]
                #####特征图计算###########
                batch_view_1_ndvi = torch.unsqueeze((batch_view_1[:,7,:,:]-batch_view_1[:,3,:,:])/(batch_view_1[:,7,:,:]+batch_view_1[:,3,:,:]),dim=1)
                batch_view_2_ndvi = torch.unsqueeze((batch_view_2[:,7,:,:]-batch_view_2[:,3,:,:])/(batch_view_2[:,7,:,:]+batch_view_2[:,3,:,:]),dim=1)
                batch_view_1_ndwi = torch.unsqueeze((batch_view_1[:,2,:,:]-batch_view_1[:,7,:,:])/(batch_view_1[:,2,:,:]+batch_view_1[:,7,:,:]),dim=1)
                batch_view_2_ndwi = torch.unsqueeze((batch_view_2[:,2,:,:]-batch_view_2[:,7,:,:])/(batch_view_2[:,2,:,:]+batch_view_2[:,7,:,:]),dim=1)
                batch_view_1_ndbsi = torch.unsqueeze((batch_view_1[:,3,:,:]-batch_view_1[:,1,:,:])/(batch_view_1[:,3,:,:]+batch_view_1[:,1,:,:]),dim=1)
                batch_view_2_ndbsi = torch.unsqueeze((batch_view_2[:,3,:,:]-batch_view_2[:,1,:,:])/(batch_view_2[:,3,:,:]+batch_view_2[:,1,:,:]),dim=1)
                # batch_view_1_ndbi = torch.unsqueeze((batch_view_1[:,7,:,:]-batch_view_1[:,11,:,:])/(batch_view_1[:,7,:,:]+batch_view_1[:,11,:,:]),dim=1)
                # batch_view_2_ndbi = torch.unsqueeze((batch_view_2[:,7,:,:]-batch_view_2[:,11,:,:])/(batch_view_2[:,7,:,:]+batch_view_2[:,11,:,:]),dim=1)

                batch_view_1_sei = torch.unsqueeze(((batch_view_1[:,0,:,:]+batch_view_1[:,8,:,:])-(batch_view_1[:,2,:,:]+batch_view_1[:,7,:,:]))/((batch_view_1[:,0,:,:]+batch_view_1[:,8,:,:])+(batch_view_1[:,2,:,:]+batch_view_1[:,7,:,:])),dim=1)
                batch_view_2_sei= torch.unsqueeze(((batch_view_2[:,0,:,:]+batch_view_2[:,8,:,:])-(batch_view_2[:,2,:,:]+batch_view_2[:,7,:,:]))/((batch_view_2[:,0,:,:]+batch_view_2[:,8,:,:])+(batch_view_2[:,2,:,:]+batch_view_2[:,7,:,:])),dim=1)

                # batch_view_1_nir = torch.unsqueeze((batch_view_1[:,7,:,:]),dim=1)
                # batch_view_2_nir = torch.unsqueeze((batch_view_2[:,7,:,:]),dim=1)

                batch_view_1_features = torch.cat([batch_view_1_ndvi,batch_view_1_ndwi,batch_view_1_ndbsi,batch_view_1_sei],dim=1)
                batch_view_2_features = torch.cat([batch_view_2_ndvi,batch_view_2_ndwi,batch_view_2_ndbsi,batch_view_2_sei],dim=1)

                # batch_view_1 = torch.cat([batch_view_1_1234,batch_view_1_features],dim=1)
                # batch_view_2 = torch.cat([batch_view_2_1234,batch_view_2_features],dim=1)
                #####特征图计算###########
                if args.loss_function == 'dino':
                    teacher_output_1 = teacher(batch_view_1[:2]) #teacher_output.shape (4,65536) # only the 2 global views pass through the teacher
                    teacher_output_2 = teacher(batch_view_2[:2])
                elif args.loss_function != 'dino':
                    # teacher_output_1 = teacher(batch_view_1) #teacher_output.shape (4,65536) # only the 2 global views pass through the teacher
                    # teacher_output_2 = teacher(batch_view_2)

                    # teacher_output_1 = teacher(batch_view_1_1234)
                    teacher_output_2 = teacher(batch_view_2_1234)
                    # teacher_output_1_features = teacher(batch_view_1_features)
                    teacher_output_2_features = teacher(batch_view_2_features)
                # 输入图像data
                # student_output_1 = student(batch_view_1)  #student_output.shape (20,65536)
                # student_output_2 = student(batch_view_2)

                student_output_1 = student(batch_view_1_1234)  #student_output.shape (20,65536)
                # student_output_2= student(batch_view_2_1234)
                student_output_1_features = student(batch_view_1_features)
                # student_output_2_features = student(batch_view_2_features)

                #计算student和teacher结果的loss
                if args.loss_function == 'dino':
                    #原dino loss
                    # loss_1 = dino_loss(student_output_1, student_output_2, epoch)
                    loss = dino_loss(student_output_1, teacher_output_2, epoch)
                    # loss_2 = dino_loss(student_output_2, teacher_output_2,epoch)
                    # loss = loss_1+loss_2
                elif args.loss_function == 'NCECriterion':
                    #NCE contrastive multiview loss
                    # loss = dino_loss(student_output_1-student_output_2, teacher_output_1-teacher_output_2,epoch)
                    loss = dino_loss(student_output_1, teacher_output_2,epoch)
                    loss += dino_loss(teacher_output_2, student_output_1,epoch)
                elif args.loss_function == 'BarlowTwins':
                    loss = dino_loss(student_output_1, teacher_output_2)
                elif args.loss_function == 'HardNegtive':
                    loss = dino_loss(student_output_1, teacher_output_2,batch_size=args.embed_dim, parameter_cca = args.cca)
                    # loss = loss + dino_loss(teacher_output_2,student_output_1,batch_size=args.embed_dim, parameter_cca = args.cca)
                    loss_features = dino_loss(student_output_1_features, teacher_output_2_features,batch_size=args.embed_dim, parameter_cca = args.cca)
                    loss = loss + 0.5*loss_features

                #########
                # loss = dino_loss(student_output_1, student_output_2, epoch)
                # loss += dino_loss(teacher_output_1, teacher_output_2, epoch)

                # loss = dino_loss(student_output_1, student_output_2)
                # loss += dino_loss(teacher_output_1, teacher_output_2)

                # if loss == float('inf'):
                #     continue
                # loss = dino_loss(torch.cat((student_output_1,student_output_2),0), torch.cat((teacher_output_1,teacher_output_2),0), epoch)
                # loss = dino_loss(student_output_1+student_output_2, teacher_output_1+teacher_output_2, epoch)
                ##########
                # loss_2 = dino_loss(student_output_2, teacher_output_2,epoch)
                # loss = loss_1
                # loss = loss_1+loss_2
                # student update
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                param_norms = None
                if fp16_scaler is None:
                    loss.backward()
                    if args.clip_grad:
                        param_norms = utils.clip_gradients(student, args.clip_grad)
                    utils.cancel_gradients_last_layer(epoch, student,
                                                      args.freeze_last_layer)
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    if args.clip_grad:
                        fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                        param_norms = utils.clip_gradients(student, args.clip_grad)
                    utils.cancel_gradients_last_layer(epoch, student,
                                                      args.freeze_last_layer)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                # EMA update for the teacher 将student参数更新至teacher
                with torch.no_grad():
                    m = momentum_schedule[it]  # momentum parameter
                    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                train_loss += loss.item()
                print('Epoch: {} \t {}/{} Training Loss: {:.6f}'.format(epoch, i+1, pbatch, train_loss))
                torch.cuda.empty_cache()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # # student update
        # optimizer.zero_grad()
        # param_norms = None
        # if fp16_scaler is None:
        #     loss.backward()
        #     if args.clip_grad:
        #         param_norms = utils.clip_gradients(student, args.clip_grad)
        #     utils.cancel_gradients_last_layer(epoch, student,
        #                                       args.freeze_last_layer)
        #     optimizer.step()
        # else:
        #     fp16_scaler.scale(loss).backward()
        #     if args.clip_grad:
        #         fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        #         param_norms = utils.clip_gradients(student, args.clip_grad)
        #     utils.cancel_gradients_last_layer(epoch, student,
        #                                       args.freeze_last_layer)
        #     fp16_scaler.step(optimizer)
        #     fp16_scaler.update()

        # # EMA update for the teacher 将student参数更新至teacher
        # with torch.no_grad():
        #     m = momentum_schedule[it]  # momentum parameter
        #     for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        #         param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging 该epoch训练结束
        torch.cuda.synchronize()
        metric_logger.update(loss=train_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        # evaluate the model
        # if (it + 1) % args.eval_freq == 0:
        #     validate(eval_loader,student,it+1)
        #     student.train()
        #
        # # save checkpoints # ============ writing logs ... 保持该epoch训练模型============
        # if (it + 1) % args.saveckp_freq == 0:
        #     save_dict = {
        #         'student': student.state_dict(),
        #         'teacher': teacher.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'epoch': epoch + 1,
        #         'args': args,
        #         'dino_loss': dino_loss.state_dict(),
        #     }
        #     if fp16_scaler is not None:
        #         save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        #     utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        #     if args.saveckp_freq and epoch % args.saveckp_freq == 0:
        #         utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                  'epoch': epoch}
        #     if utils.is_main_process():
        #         with (Path(args.output_dir) / "log.txt").open("a") as f:
        #             f.write(json.dumps(log_stats) + "\n")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def patchize(img: torch.Tensor, patch_size, unfold_stride) -> torch.Tensor:

        """
        img.shape
        B  : batch size
        C  : channels of image (same to patches.shape[1])
        iH : height of image
        iW : width of image

        pH : height of patch
        pW : width of patch
        V  : values in a patch (pH * pW * C)
        """

        B, C, iH, iW = img.shape
        pH = patch_size
        pW = patch_size

        unfold = nn.Unfold(kernel_size=(pH, pW), stride=unfold_stride)

        patches = unfold(img)  # (B, V, P)
        patches = patches.permute(0, 2, 1).contiguous()  # (B, P, V)
        patches = patches.view(-1, C, pH, pW)  # (P, C, pH, pW)
        return patches

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
# empirical cross-correlation matrix
class BarlowTwins_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # projector
        # sizes = [2048] + list(map(int, args.projector.split('-')))
        # layers = []
        # for i in range(len(sizes) - 2):
        #     layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        #     layers.append(nn.BatchNorm1d(sizes[i + 1]))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        # self.projector = nn.Sequential(*layers)
        #
        # # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(args.out_dim, affine=False)

    def forward(self, y1, y2):

        # empirical cross-correlation matrix
        c = self.bn(y1).T @ self.bn(y2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def get_corr(fake_Y, Y):#计算两个向量person相关系数
        fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
        fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
        corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
                    torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
        return corr

class HardNegtive_loss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data,eps,ncrops,out_dim,teacher_temp,warmup_teacher_temp,warmup_teacher_temp_epochs,nepochs,student_temp=0.1,center_momentum=0.9):
        super(HardNegtive_loss, self).__init__()
        self.n_data = n_data
        self.eps = eps
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    def forward(self, out_11, out_22, tau_plus=0.1, batch_size=500, beta=1.0, temperature=0.5, estimator='hard',parameter_cca = 0.1):   #x.shape (2,16385,1)
        #归一化)
        # batch_size = 100
        # out_1 = out_1[0:100,:]
        # out_2 = out_2[0:100,:]
        # df.iloc[[1, 3]]
        #####CCA#####特征选择V1
        # import copy
        # out_1_cpu = copy.deepcopy(out_11.T.cpu().detach().numpy())
        # out_2_cpu = copy.deepcopy(out_22.T.cpu().detach().numpy())
        # out_1_cpu[out_1_cpu == 0] += 0.00001
        # out_2_cpu[out_2_cpu == 0] += 0.00001
        # cca = CCA(n_components=batch_size,max_iter=10)
        # cca.fit(out_1_cpu, out_2_cpu)
        # X_train_r, Y_train_r = cca.transform(out_1_cpu, out_2_cpu)
        #
        # index = []
        # for i in range(0,batch_size):
        #     rr = np.corrcoef(X_train_r[:, i], Y_train_r[:, i])[0, 1]
        #     if rr < parameter_cca: #设置选择样本的相关系数的阈值
        #         index.append(i)
        #
        # indices = torch.tensor(index).cuda()
        # out_1 = torch.index_select(out_11, 0, indices)
        # out_2 = torch.index_select(out_22, 0, indices)
        # batch_size = out_1.shape[0]
        # #####CCA#####特征选择V1

        # #####相关系数#####特征选择V2
        # index = []
        # for i in range(0,batch_size):
        #     rr = get_corr(out_11[i, :], out_22[i, :])
        #     if rr < parameter_cca: #设置选择样本的相关系数的阈值
        #         index.append(i)
        # indices = torch.tensor(index).cuda()
        # out_1 = torch.index_select(out_11, 0, indices)
        # out_2 = torch.index_select(out_22, 0, indices)
        # batch_size = out_1.shape[0]
        # #####相关系数#####特征选择V2

        # #####相关系数#####特征选择V3  这精度有提升
        # from itertools import groupby
        # rr = torch.cosine_similarity(out_11, out_22, dim=0)
        # rr_numbers = []
        # rr_cont = []
        # for k, g in groupby(sorted(rr), key=lambda x: x//0.01):
        #     aa = len(list(g))
        #     rr_cont.append(aa)
        #     rr_numbers.append([k*0.01,(k+1)*0.01,aa])
        #     # print('{}-{}: {}'.format(k*0.001, (k+1)*0.001, len(list(g))))
        # cont_mean = np.mean(rr_cont)
        # index = []
        # for i in range(0,len(rr_numbers)):
        #     if rr_numbers[i][2] > cont_mean:
        #        for j in range(0,len(rr)):
        #            if rr[j] >= rr_numbers[i][0] and rr[j] < rr_numbers[i][1]:
        #               index.append(j)
        # indices = torch.tensor(index).cuda()
        # out_1 = torch.index_select(out_11, 1, indices)
        # out_2 = torch.index_select(out_22, 1, indices)
        # self.center_update = torch.index_select(self.center, 1, indices)
        # #####相关系数#####特征选择V3   这精度有提升

        # #####相关系数#####特征选择V4
        rr = torch.cosine_similarity(out_11, out_22, dim=0)
        index = []
        for i in range(0,len(rr)):
            if rr[i] > parameter_cca:
                index.append(i)
        indices = torch.tensor(index).cuda()
        out_1 = torch.index_select(out_11, 1, indices)
        out_2 = torch.index_select(out_22, 1, indices)
        self.center_update = torch.index_select(self.center, 1, indices)
        # #####相关系数#####特征选择V4

        # out_1 = out_11
        # out_2 = out_22

        out_1 = out_1 / self.student_temp #out_1: 384,6000
        # out_2 = F.softmax((out_2 - self.center) / temperature, dim=-1)
        out_2 = F.softmax((out_2 - self.center_update) / temperature, dim=-1) #####相关系数#####特征选择V3
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)
        # neg score
        out = torch.cat([out_1, out_2], dim=0)

        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature) #公式1，a.contiguous(): 改变a.contiguous()的值，a不受影响
        # old_neg = neg.clone()
        # 采样负样本的方法，不同位置的样本为负样本，可以从这里做创新，结合先验知识类别，寻找hard samples
        mask = get_negative_mask(batch_size).cuda() #mask: 768,768
        neg = neg.masked_select(mask).view(2 * batch_size, -1) #neg: 768,768 -> 768,766

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature) #公式2
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (beta * neg.log()).exp() #公式3; imp:768,766; neg:768,766; neg.log():以e为底求neg的对数; a.exp(): 求e的a次方
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1) #公式3; reweight_neg: 768; (imp * neg).sum(dim=-1): 768, 对行求和; .mean(dim=-1): 768, 对行求平均值
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus) #公式3; pos: 768, 正样本; Ng: 768
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))  #公式4; torch.clamp(a,min=1): 将小于1的数提换为1
        elif estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean() #torch.log: 以e为底求neg的对数
        # loss.requires_grad_(True)
        # print(loss)
        # self.update_center(out_22)
        self.update_center(out_2)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        # self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center_update = self.center_update * self.center_momentum + batch_center * (1 - self.center_momentum) #####相关系数#####特征选择V3

# class contrastive_loss(nn.Module):
class NCECriterion_loss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data,eps,ncrops,out_dim,teacher_temp,warmup_teacher_temp,warmup_teacher_temp_epochs,nepochs,student_temp=0.1,center_momentum=0.9):
        super(NCECriterion_loss, self).__init__()
        self.n_data = n_data
        self.eps = eps
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output,epoch):   #x.shape (2,16385,1)
        #归一化)

        student_out = student_output / self.student_temp
        # student_out = F.softmax(student_output / self.student_temp, dim=-1)

        # student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        # teacher_out = teacher_output / temp
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # teacher_out = teacher_output
        # teacher_out = teacher_out.detach().chunk(2)

        x = torch.cat((student_out, teacher_out), 0)
        xx = torch.cat((student_out, teacher_out), 0)
        for i in range(0,x.shape[0]):
            x[i] = (xx[i] - torch.min(xx[i])) / (torch.max(xx[i])-torch.min(xx[i]))
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0) #每张图取一个正样本
        #torch.div(a,b):a/b   .log_(a):以e为底数求a的对数  a.add(b):对a所有元素+b
        log_D1 = torch.div(P_pos+ self.eps, P_pos.add(m * Pn + self.eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m) #每张图取前一个正样本以外的为负样本
        #如果需要保存旧的tensor即需要开辟新的存储地址而不是引用，可以用 clone() 进行深拷贝
        #a.fill_(b) 将a所有元素值用b替换
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + self.eps)).log_()
            #a.view(-1,1)在函数的参数中经常可以看到-1例如x.view(-1, 4)
            #这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成4列，那不确定的地方就可以写成-1
            #对sum()的说明
            # a = torch.ones((2, 3))
            # print(a)
            # tensor([[1., 1., 1.],
            #        [1., 1., 1.]])
            # a1 = torch.sum(a, dim=(0, 1))
            # a2 = torch.sum(a, dim=0)
            # a3 = torch.sum(a, dim=1)
            # print(a1)
            # print(a2)
            # print(a3)
            # tensor(6.)
            # tensor([2., 2., 2.])
            # tensor([3., 3.])
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)[0]) / bsz
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self, n_data,eps,ncrops,out_dim,teacher_temp,warmup_teacher_temp,warmup_teacher_temp_epochs,nepochs,student_temp=0.1,center_momentum=0.9):
        super(Similarity, self).__init__()
        self.n_data = n_data
        self.eps = eps
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, g_s, g_t,epoch):

        return self.similarity_loss(g_s, g_t,epoch)

    def similarity_loss(self, student_output, teacher_output,epoch):
        f_s = student_output / self.student_temp
        temp = self.teacher_temp_schedule[epoch]
        f_t = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # f_s = student_output
        # f_t = teacher_output

        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        self.update_center(f_t)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    # path_files = 'brasilia,montpellier,norcia,rio,saclay_w,valencia,dubai,lasvegas,milano,chongqing'
    path_files = 'dubai,lasvegas,milano,chongqing'
    # path_files = 'norcia,saclay_w,valencia,dubai,lasvegas,milano' #精度比chen差的站点
    # path_files = 'brasilia,dubai,milano,rio,' #可增大epoch继续训练的站点

    path_list = path_files.split(',')
    for i in range(0,len(path_list)):
        parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
        args = parser.parse_args()
        args.output_dir = 'G:/test_dino/paper_v2_dino_change_detection/oscd_one_img_train_paper_v2_discussion_PK_4_fusion_laptop/'+path_list[i]
        args.data_dir_train = 'E:/research_doctoral_laptop/data/change_detection/OSCD/train_1_image/'+path_list[i]
        args.data_dir_eval = 'E:/research_doctoral_laptop/data/change_detection/OSCD/train_1_image/'+path_list[i]
        args.oscd_data_name = path_list[i]
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_dino(args,i)
