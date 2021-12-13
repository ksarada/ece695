#!/bin/bash

type="wb"
source="snnbp" #"ann"
target="snnbp" #"snnconv" #"ann"
filename="evaluate_attacks.py"
#FGSM attack
#*************
#CUDA_VISIBLE_DEVICES=0 python $filename --arch 'VGG5' --dataset 'CIFAR10' \
#--attack 'fgsm' --type $type --source $source --target $target --epsilon 8 --batch_size 4

#PGD attack
#************
CUDA_VISIBLE_DEVICES=2 python3 $filename --arch 'VGG11' --dataset 'CIFAR10' \
--attack 'pgd' --type $type --source $source --target $target --epsilon 32 \
--eps_iter 1 --pgd_steps 10 --batch_size 32 --rand_init 0 --targeted 'True' --num_batches 1


