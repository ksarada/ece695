import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
import torch.optim.lr_scheduler as sched
import time

def train_proxy(model_source, model_target, lr, wd, momentum):

    batch_size = 256
    criterion = nn.CrossEntropyLoss().cuda(0) #args.gpu)
    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model_target.parameters()), lr,
                                momentum=momentum, weight_decay=wd, nesterov=True)
    scheduler = sched.MultiStepLR(optimizer, gamma=0.5, milestones=range(10, 300, 10))

    normalize       = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

    T = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.RandomCrop(32,4),
			
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       
		])

    train_sampler = None
    cifar10 = datasets.CIFAR10(root='/home/min/a/skrithiv/Hebb_l/data1', train=True, download=True, transform=T)
    train_loader = torch.utils.data.DataLoader(
        cifar10, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=10, pin_memory=True, sampler=train_sampler)
    
    
    Ttest = transforms.Compose([transforms.ToTensor(), normalize]) 
    cifar10 = datasets.CIFAR10(root='/home/min/a/skrithiv/Hebb_l/data', train=False, download=True, transform=Ttest)
    val_loader = DataLoader(cifar10, batch_size=batch_size, num_workers=10)

    for epoch in range(0, 90):
        adjust_learning_rate(optimizer, epoch, lr)
        train(train_loader, model_target, model_source, criterion, optimizer, epoch) #, args)
        # train for one epoch

        # evaluate on validation set
        acc1 = validate(val_loader, model_target, criterion) #, args)

        # remember best acc@1 and save checkpoint
        #is_best = acc1 > best_acc1
        #best_acc1 = max(acc1, best_acc1)

    return model       

def train(train_loader, model_target, model_source, criterion, optimizer, epoch): #, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model_target.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        if(i>1):
            data_time.update(time.time() - end)
        #end1 = time.time()
        if 1: #args.gpu is not None:
            images = images.cuda(0, non_blocking=True)
        target = target.cuda(0, non_blocking=True)
        end1 = time.time()
        # compute output
        #if scheduler is not None: scheduler.step()
        #output, ActInt = model(images)
        output, _, _ = model_target(images) #, 1. ) #lr_const)
        loss = criterion(output, target) #+ 0.01*loss_comp

        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #model.layer1[0].c1_out.retain_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # measure elapsed time
        if(i>1):
            batch_time.update(time.time() - end1)
        end = time.time()

        if i % 100 == 0:
            progress.display(i)
    

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if 1: #args.gpu is not None:
                images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            # compute output
            #output, actOut = model(images)
            output, _, _ = model(images) #, 1.)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0: #args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.5 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
       
    
