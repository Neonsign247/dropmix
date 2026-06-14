#!/bin/bash
NUM_PROC=$1
DROPMIX_RATE=$2
shift
echo CUDA_VISIBLE_DEVICES=$NUM_PROC 
export CUDA_VISIBLE_DEVICES=$NUM_PROC
echo $DROPMIX_RATE
for i in 2
do
    python main.py --dataset cifar100 --labels_per_class 500 --arch wrn28_10 \
        --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 300 \
        --schedule 100 200 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 \
        --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8 \
        --dropmix_rate $DROPMIX_RATE --seed $i
done
