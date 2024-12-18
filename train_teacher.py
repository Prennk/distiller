from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar100 import get_upsampled_cifar100_dataloaders, get_upsampled_cifar100_dataloaders_sample
from dataset.roadsign import get_road_sign_dataloaders
from dataset.oxford_flowers import get_flowers102_dataloader

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--resume', action='store_true', help='resume train')
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint path')
    parser.add_argument('--pretrained', action='store_true', help='pretrained')
    parser.add_argument('--pretrained_path', type=str, default='', help='pretrained path')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'road_sign', 'flowers'], help='dataset')
    parser.add_argument('--upsample', action='store_true', help='upsample 416x416')

    # model
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'mobilenetv2_6_05', 'mobilenetv2_6_1', 'mobilenetv2_6_025', 'mobilenetv2_half_backbone',
                                 'ShuffleV1', 'ShuffleV2', 
                                 'darknet19', 'darknet53', 'darknet53e', 'cspdarknet53', 'cspdarknet53_backbone',
                                 'efficientnet_b0',
                                 'repvit_m0_6', 'repvit_m0_9', 'repvit_m1_0', 'repvit_m1_1', 'repvit_m1_5', 'repvit_m2_3'])

    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--note', type=str, default='experiment_1', help='the experiment note')

    opt = parser.parse_args()
    
    torch.manual_seed(opt.seed)

    # set different learning rate from these 4 models
    if opt.model in ['mobilenetv2_6_025', 'mobilenetv2_6_05', 'mobilenetv2_6_1', 'mobilenetv2_half_backbone', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.upsample:
        opt.model_name = '{}_UPSAMPLE_{}_lr_{}_decay_{}_note_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                                    opt.weight_decay, opt.note)
    else:
        opt.model_name = '{}_{}_lr_{}_decay_{}_note_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                                    opt.weight_decay, opt.note)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()
    torch.manual_seed(opt.seed)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.upsample:
            train_loader, val_loader = get_upsampled_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 100
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
            n_cls = 100
    elif opt.dataset == 'road_sign':
        train_loader, val_loader = get_road_sign_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 8
    elif opt.dataset == 'flowers':
        train_loader, val_loader = get_flowers102_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 102
    else:
        raise NotImplementedError(opt.dataset)

    torch.manual_seed(opt.seed)
    
    # model
    model = model_dict[opt.model](num_classes=n_cls)

    if opt.pretrained and not opt.resume:
        print(f"Loading pretrained model from {str(opt.pretrained_path)} ...")
        pretrained_model_state_dict = torch.load(opt.pretrained_path)
        model_state_dict = model.state_dict()

        # Check which keys are matched and which are not
        matched_keys = []
        unmatched_keys = []

        for k, v in pretrained_model_state_dict.items():
            if k in model_state_dict:
                matched_keys.append(k)
            else:
                unmatched_keys.append(k)

        print("Count Matched keys:", len(matched_keys))
        print("Matched keys:", matched_keys)
        print("\nCount Unmatched keys:", len(unmatched_keys))
        print("Unmatched keys:", unmatched_keys)

        # Load only matched keys
        model.load_state_dict(pretrained_model_state_dict, strict=False)

        print("Only matched keys are loaded")
        print("Pretrained model loaded successfully")


    if opt.resume:
        print(f"Loading checkpoint from {str(opt.checkpoint_path)} ...")

        model.load_state_dict(torch.load(str(opt.checkpoint_path))['model'])
        current_epoch = torch.load(opt.checkpoint_path)['epoch'] + 1

        print(f"Checkpoint loaded")

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(current_epoch if opt.resume else 1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)


        # create checkpoint
        print(f'==> Checkpoint created for epoch {epoch}...')
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'accuracy': test_acc,
            'optimizer': optimizer.state_dict(),
        }
        save_file = os.path.join(opt.save_folder, 'checkpoint.pth'.format(epoch=epoch))
        torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
