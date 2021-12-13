import argparse
import os
import random
import shutil
import time
import warnings

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
import xlwt 
from xlwt import Workbook 
import nn_models as vgg
import model_loader

from torch.utils.data.sampler import SubsetRandomSampler

model_list = {
    'ann'              : vgg.ANN_VGG,
    'ann_source'       : vgg.ANN_VGG,
    'ann_target'       : vgg.ANN_VGG,
    'snnconv'          : vgg.SNN_VGG,
    'snnbp'            : vgg.SNN_VGG,
}


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sf', 
                    help='path to store checkpoint of current epoch')
parser.add_argument('--bsf', 
                    help='path to store best_acc checkpoint')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if 1: #args.seed is not None:
        random.seed(0) #args.seed)
        torch.manual_seed(0) #args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    snn_thresholds = [[5.9, 1.77, 0.78, 0.64, 0.24, 1.02, 1.45, 3.13, 1.04], [5.90, 1.77, 0.78, 0.64, 0.22, 1.02, 1.4, 3.13, 1.04]]

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        #model = model_list['ann'](vgg_name = 'VGG11', labels = 10) #.cuda() #models.__dict__[args.arch]()
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        #model = models.__dict__[args.arch]()
        #model = model_loader.load('ann', 32, '/home/min/a/skrithiv/AdvPowerAttacks/hybrid-snn-conversion-master/vgg11_mtsnn.pth', snn_thresholds[0][0:])
        model_file = '/home/min/a/skrithiv/AdvPowerAttacks/hybrid-snn-conversion-master/vgg11_mtsnn.pth'
        
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        cur_dict = model.state_dict() 

        kIn = 0
        lIn = 0
        for k, v in stored['state_dict'].items():
        #if(kIn<14):
            #print(k)
            if(k.find('weight')!=-1):
                jIn = 0
                for k1, v1 in cur_dict.items():
                    if(jIn==kIn):
                        cur_dict[k1] = nn.Parameter(stored['state_dict'][k].data)
                        lIn = lIn + 1
                        #print(kIn,end='')
                        #print(jIn,end='')
                        #print(k1,end='')
                        #print(' ', end="")
                        #print(k)
                    jIn = jIn + 1
                kIn = kIn + 1
    
        model.load_state_dict(cur_dict)
          

    '''
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    '''

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

 

    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), args.lr) #,
                                #weight_decay=args.weight_decay) #, nesterov=True)

    scheduler = sched.MultiStepLR(optimizer, gamma=0.5, milestones=range(10, 300, 10))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = 0.84 #checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True #True #False #True #False #True
    
    #cifar10 = CIFAR10(root='/home/min/a/skrithiv/Hebb_l/data', train=True, download=True)
    #X = cifar10.data[0:50000] 
    #X = X / 255.
    #mean = X.mean(axis=(0, 1, 2), keepdims=True)
    #print(mean)
    #std = X.std(axis=(0, 1, 2), keepdims=True)
    #print(std)

    mean = [129.30416561, 124.0699627,  112.43405006]
    std = [68.1702429,  65.39180804, 70.41837019]

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    '''
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    '''

    T = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.RandomCrop(32,4),
			
			transforms.ToTensor(),
			#transforms.Normalize(mean , std)
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
                        #transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
		])
		
    Ttest = transforms.Compose([
			transforms.ToTensor(),
			#transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			#transforms.Normalize(mean , std)
                        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
                        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

    cifar10 = CIFAR10(root='/home/min/a/skrithiv/Hebb_l/data1', train=True, download=True, transform=T)

    #if args.distributed:
    #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #else:
    train_sampler = SubsetRandomSampler(range(50000)) #None
    

    '''   
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    '''

    train_loader = DataLoader(cifar10, batch_size=args.batch_size, sampler = train_sampler, num_workers = args.workers)
    
    '''
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''
    cifar10 = CIFAR10(root='/home/min/a/skrithiv/Hebb_l/data', train=False, download=True, transform=Ttest)
    #return DataLoader(cifar10, batch_size=self.BATCH_SIZE, num_workers=P.NUM_WORKERS)
    val_loader = DataLoader(cifar10, batch_size=args.batch_size, num_workers=args.workers)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
 
    #snn_thresholds = [[5.9, 1.77, 0.78, 0.64, 0.24, 1.02, 1.45, 3.13, 1.04], [5.90, 1.77, 0.78, 0.64, 0.22, 1.02, 1.4, 3.13, 1.04]]
    model_source = model_loader.load('snnbp', 32, '/home/min/a/skrithiv/AdvPowerAttacks/hybrid-snn-conversion-master/vgg11_mtsnn.pth', snn_thresholds[0][0:])
    #model_source.features = torch.nn.DataParallel(model_source.features)
    model_source.cuda()
    # Workbook is created 
    #wb = Workbook() 
    #sheet1 = wb.add_sheet('Sheet_Error_VAR') 

    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
         #   train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)


        train(train_loader, model, model_source, criterion, optimizer, scheduler, epoch, args)
        # train for one epoch

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best) #, args.sf, args.bsf) #'r50_hybrid.pth.tar')

    #wb.save('Cifar10_bl_sgd.xls')


def train(train_loader, model, model_source, criterion, optimizer, scheduler, epoch, args):
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
    model.train()
    model_source.eval()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        if(i>1):
            data_time.update(time.time() - end)
        #end1 = time.time()
        if 1: #args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        end1 = time.time()
        # compute output
        #if scheduler is not None: scheduler.step()
        #output, ActInt = model(images)

        '''
        avg_sc = {}
        output_spike, input_sc, gm_conv1, obj_count, total_sc, input_tsc  = model_source(images, 0, 40, True)
        lIn = 0
        for k, val in total_sc.items():
            avg_sc[lIn] = torch.mean(val, 0, True)
            #print(avg_sc[lIn].size())
            lIn = lIn + 1
        '''
        avg_act = {}
        output, act_conv = model(images) #, 1. ) #lr_const)
        lIn = 0
        
        for k, val in act_conv.items():
            avg_act[lIn] = val #torch.mean(val, 0, True)
            #print(avg_act[lIn].size())
            lIn = lIn + 1

        clean_sc = 0
        lIn = 0

        for k,val in act_conv.items(): #avg_act.items():
            #print('{:.2f} {:.2f}'.format(torch.sum(avg_act[lIn]).item(), torch.sum(avg_sc[lIn]).item()))
            if(lIn > 0):
                clean_sc = clean_sc + torch.norm(avg_act[lIn]) #(avg_act[lIn] - avg_sc[lIn])**2.)
            lIn = lIn + 1
        
     
        loss1 = criterion(output, target) 
        loss2 = clean_sc
        loss = loss1 #+ 0.001*loss2 

        #print(' {:.3f}'.format(loss.item()))   
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.cuda(), target, topk=(1, 5))
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

        if i % args.print_freq == 0:
            progress.display(i)
    #return torch.norm(gradNorm_L2avg)
    #return grad_er_var, grad_er_mean, grad_er_max
    #return losses.avg
        
def validate(val_loader, model, criterion, args):
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
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            #output, actOut = model(images)
            output, _ = model(images) #, 1.)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename = 'best.pth.tar'):
    #for k, v in state.items():
    #    print(k)
    torch.save(state, 'vgg_models/vgg5_2.pth.tar') #filename)
    #if is_best:
    #    shutil.copyfile(filename, 'vgg_models/vgg5_b.pth.tar') #best_filename) #'r50_hybrid_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 20))
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


if __name__ == '__main__':
    main()

