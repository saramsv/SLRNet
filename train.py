# System libs
import os
import sys
import time
# import math
import random
import argparse
import shutil
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset, DecomTrainDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback

import evaluate
from itertools import cycle
import distribution
from pytorch_memlab import MemReporter
 

# train one epoch
def cal_loss(segmentation_module, batch_data):
    # forward pass
    loss, acc = segmentation_module(batch_data) #, sup_batch_data)
    loss = loss.mean()
    acc = acc.mean()
    return loss, acc

def train(segmentation_module, iterator, optimizers, history, epoch, cfg): #, sup_iterator=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        #sup_batch_data = None
        #if sup_iterator !=None:
        #    sup_batch_data = next(sup_iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        #loss, acc = cal_loss(segmentation_module, batch_data)
        loss, acc = segmentation_module(batch_data) #, sup_batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())

        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        #sys.stdout.flush()


        ## Added them because of the cuda out of memory error
        #reporter = MemReporter()
        #reporter.report()
        del loss
        torch.cuda.empty_cache()


def checkpoint(nets, history, cfg, epoch, is_best):
    # I am using 0 in the name of the current model
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, 0))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, 0))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, 0))
    if is_best:
        shutil.copyfile('{}/history_epoch_{}.pth'.format(cfg.DIR, 0), 
        '{}/history_epoch_best.pth'.format(cfg.DIR))
        shutil.copyfile('{}/encoder_epoch_{}.pth'.format(cfg.DIR, 0), 
        '{}/encoder_epoch_best.pth'.format(cfg.DIR))
        shutil.copyfile('{}/decoder_epoch_{}.pth'.format(cfg.DIR, 0),
        '{}/decoder_epoch_best.pth'.format(cfg.DIR))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)

    #nSamples = [611, 648, 754, 1169, 591, 429] #, 287]
    '''
    train_annotations_colored_list = distribution.get_paths(os.path.join(cfg.DATASET.root_dataset, cfg.DATASET.list_train))
    if cfg.TRAIN.sup == True:
        train_annotations_colored_list.extend(distribution.get_paths(os.path.join(cfg.DATASET.root_dataset, cfg.DATASET.list_sup_train)))
    # nSamples = distribution.main(train_annotations_colored_list)[1:]
    nSamples = distribution.main(train_annotations_colored_list, bg=False)
    normedWeights = [1 - x for x in nSamples]#[1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).cuda()

    crit = nn.NLLLoss(ignore_index=-1, weight=normedWeights) 
    '''
    crit = nn.NLLLoss(ignore_index=-1) 

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.batch_size_per_gpu, cfg.TRAIN.type, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.batch_size, cfg.TRAIN.type)

    # Supervised dataset and Loader
    if cfg.TRAIN.sup == True:
        dataset_sup_train = TrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_sup_train,
            cfg.DATASET,
            batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

        loader_sup_train = torch.utils.data.DataLoader(
            dataset_sup_train,
            batch_size=len(gpus),  # we have modified data_parallel
            shuffle=False,  # we do not use this param
            collate_fn=user_scattered_collate,
            num_workers=cfg.TRAIN.workers,
            drop_last=True,
            pin_memory=True)

        # create sup loader iterator
        iterator_sup_train = iter(loader_sup_train)

    # Dataset and Loader
    if cfg.TRAIN.type == 'seq':
        dataset_train = DecomTrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_train,
            cfg.DATASET,
            batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
    else:
        dataset_train = TrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_train,
            cfg.DATASET,
            batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)

    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    if cfg.TRAIN.sup == True:
        if len(loader_sup_train) < len(loader_train):
            #iterator_train = iter(zip(cycle(loader_sup_train), loader_train))
            iterator_train = iter(zip(loader_sup_train, loader_train))
        else:
            #iterator_train = iter(zip(loader_sup_train, cycle(loader_train)))
            iterator_train = iter(zip(loader_sup_train, loader_train))

    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}
    best_acc = 0
    best_IoU = 0
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg) #, iterator_sup_train) 
        
        #checkpoint(nets, history, cfg, epoch+1, False)
        if epoch > 0:
            cfg.MODEL.weights_encoder = os.path.join(
                cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
            cfg.MODEL.weights_decoder = os.path.join(
                cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
        #if epoch % 2 == 0: # do the evaluation on the validate data every other epoch
        with torch.no_grad():
            current_IoU, current_acc = evaluate.main(cfg, 0)
    
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        best_IoU = max(current_IoU, best_IoU)
        # checkpointing
        checkpoint(nets, history, cfg, epoch+1, is_best)
        #if is_best:
        print("Epoch: {}, Current best IoU: {}, current best acc: {}".format(epoch+1, best_IoU, best_acc))
        

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    #if cfg.TRAIN.start_epoch > 0:
    if cfg.TRAIN.start_epoch != 0:
        cfg.MODEL.weights_encoder = os.path.join(
            #cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
            cfg.DIR, 'encoder_epoch_best.pth')
        cfg.MODEL.weights_decoder = os.path.join(
            #cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
            cfg.DIR, 'decoder_epoch_best.pth')
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder
    cfg.TRAIN.type = cfg.TRAIN.type

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
