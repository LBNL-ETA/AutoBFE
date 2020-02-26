#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python predict.py --backbone resnet \
 --workers 8 \
 --test-batch-size 16 --gpu-ids 0 --best_model 'model_best.pth.tar' \
 --test_path '/Data/test/' #\
