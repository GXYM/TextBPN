import os
import gc
import time
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

from dataset import SynthText, TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import Augmentation
from cfglib.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from cfglib.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary
from util.shedule import FixLR
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)

lr = None
train_step = 0


def save_model(model, epoch, lr, optimzer):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'TextBPN_{}_{}.pth'.format(model.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict()
        # 'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def _parse_data(inputs):
    input_dict = {}
    inputs = list(map(lambda x: to_device(x), inputs))
    input_dict['img'] = inputs[0]
    input_dict['train_mask'] = inputs[1]
    input_dict['tr_mask'] = inputs[2]
    input_dict['distance_field'] = inputs[3]
    input_dict['direction_field'] = inputs[4]
    input_dict['weight_matrix'] = inputs[5]
    input_dict['gt_points'] = inputs[6]
    input_dict['proposal_points'] = inputs[7]
    input_dict['ignore_tags'] = inputs[8]

    return input_dict


def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    # scheduler.step()

    print('Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()))

    for i, inputs in enumerate(train_loader):

        data_time.update(time.time() - end)
        train_step += 1
        input_dict = _parse_data(inputs)
        output_dict = model(input_dict)
        loss_dict = criterion(input_dict, output_dict, eps=epoch)
        loss = loss_dict["total_loss"]
        # backward
        try:
            optimizer.zero_grad()
            loss.backward()
        except:
            print("loss gg")
            continue
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        gc.collect()

        if cfg.viz and i % cfg.viz_freq == 0 and i > 0 and epoch % 2 == 0:
            visualize_network_output(output_dict, input_dict, mode='train')

        if i % cfg.display_freq == 0:
            print_inform = "({:d} / {:d}) ".format(i, len(train_loader))
            for (k, v) in loss_dict.items():
                print_inform += " {}: {:.4f} ".format(k, v.item())
            print(print_inform)

    if epoch % cfg.save_freq == 0 and epoch > 200:
        save_model(model, epoch, scheduler.get_lr(), optimizer)

    print('Training Loss: {}'.format(losses.avg))


def main():

    global lr
    if cfg.exp_name == 'Totaltext':
        trainset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        # valset = TotalText(
        #     data_root='data/total-text-mat',
        #     ignore_list=None,
        #     is_training=False,
        #     transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        # )
        valset = None

    elif cfg.exp_name == 'Synthtext':
        trainset = SynthText(
            data_root='data/SynthText',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Ctw1500':
        trainset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Icdar2015':
        trainset = Icdar15Text(
            data_root='data/Icdar2015',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    elif cfg.exp_name == 'MLT2017':
        trainset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'TD500':
        trainset = TD500Text(
            data_root='data/TD500',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    else:
        print("dataset name is not correct")

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    # Model
    model = TextNet(backbone=cfg.net, is_training=True)
    if cfg.mgpu:
        model = nn.DataParallel(model)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, cfg.resume)

    criterion = TextLoss()

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == 'Synthtext':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == 'Synthtext':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    print('Start training TextGraph.')
    for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.max_epoch+1):
        scheduler.step()
        train(model, train_loader, criterion, scheduler, optimizer, epoch)

    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()

