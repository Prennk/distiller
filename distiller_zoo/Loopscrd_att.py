from __future__ import print_function, division
import sys
import time
import torch

from .util import AverageMeter, accuracy

def move_to_device(data):
    """Move data to the appropriate device."""
    if torch.cuda.is_available():
        return data.cuda()
    return data

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """Vanilla training loop."""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = move_to_device(input.float())
        target = move_to_device(target)

        # Forward pass
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update meters
        batch_time.update(time.time() - end)
        end = time.time()

        # Print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, losses.avg
    
def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    for module in module_list:
        module.train()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd_crd = criterion_list[2] if opt.distill in ['crd', 'crd_attention'] else None
    criterion_kd_attention = criterion_list[3] if opt.distill == 'crd_attention' else None
    #criterion_kd_hinton = criterion_list[3] if opt.distill == 'crd_attention' else None

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        # Memproses input data dan menangani contrast_idx jika ada
        if opt.distill == 'crd_attention':
            if len(data) == 4:
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
                contrast_idx = None  # Set ke None jika tidak disediakan
        else:
            input, target, index = data
            contrast_idx = None
    
        data_time.update(time.time() - end)
        input = input.float()
        
        if torch.cuda.is_available():
            input, target, index = input.cuda(), target.cuda(), index.cuda()
            if contrast_idx is not None:
                contrast_idx = contrast_idx.cuda()
    
        # Forward pass
        preact = opt.distill == 'abound'
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
    
        # Menghitung primary loss dan divergence loss
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
    
        # CRD loss
        loss_kd_crd = 0  # Inisialisasi
        if criterion_kd_crd is not None:
            f_s, f_t = feat_s[-1], feat_t[-1]
            loss_kd_crd = criterion_kd_crd(f_s, f_t, index, contrast_idx) if contrast_idx is not None else criterion_kd_crd(f_s, f_t, index)
    
        # Attention loss
        loss_kd_attention = 0
        if criterion_kd_attention is not None:
            g_s, g_t = feat_s[1:-1], feat_t[1:-1]
            loss_kd_attention = sum(criterion_kd_attention(g_s_i, g_t_i) for g_s_i, g_t_i in zip(g_s, g_t)) / len(g_s)
    
        # Hinton loss
        #loss_kd_hinton = criterion_kd_hinton(logit_s, logit_t)
    
        # Menggabungkan seluruh loss dengan bobot masing-masing
        loss = (opt.gamma * loss_cls +
                opt.alpha * loss_div +
                opt.beta * (loss_kd_crd + opt.delta * loss_kd_attention))

        #loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_crd + opt.delta * loss_kd_hinton
    
        # Menghitung akurasi dan memperbarui metrik
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
    
        # Backward pass dan langkah optimasi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Memperbarui waktu batch
        batch_time.update(time.time() - end)
        end = time.time()


        # Log progress
        if idx % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    return top1.avg, losses.avg



def validate(val_loader, model, criterion, opt):
    """Validation loop."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):
            input = move_to_device(input.float())
            target = move_to_device(target)

            # Compute output
            output = model(input)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
