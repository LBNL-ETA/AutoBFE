#!/bin/bash

 CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet  \
 --lr 0.0001 --workers 6 --epochs 160 \
 --batch-size 16 --gpu-ids 0 --experiment SF_test --eval-interval 1 --dataset building \
 --loss-type dicewce \
 --optim Adam \
 --sigma 5 \
 --w0 10 \
 --train_path '/Data/train/' \
 --val_path '/Data/val/' 