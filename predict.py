import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import csv

from PIL import Image
from tqdm import tqdm
from modeling.dataloaders import building
from modeling.deeplab.deeplab_model import *
from modeling.utils.loss import SegmentationLosses
from modeling.utils.metrics import Evaluator
from common.colors import continuous_palette_for_color



class Predict(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloaders
        self.test_loader = building.get_loader(image_path=args.test_path,
                                    image_size=512,
                                    batch_size=args.test_batch_size,
                                    num_workers=args.workers,
                                    data_type='test',
                                    augment_prob=0.,
                                    weighted_loss_function=0)

        self.nclass = 2

        # Define network
        self.model = DeepLab(num_classes=self.nclass,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        
        # Using cuda
        if args.cuda:
            #self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            #patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Loading the best model
        if not os.path.isfile(args.best_model):
            raise RuntimeError("=> no best model found at '{}'" .format(args.best_model))
        best_model = torch.load(args.best_model)
        self.model.load_state_dict(best_model['state_dict'])
        self.model.train(False)
        self.model.eval()

    def predict(self):
        result_path = os.path.split(self.args.best_model)[0]

        tbar = tqdm(self.test_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image, target, tile_zxy = sample['image'], sample['label'], sample['tile_zxy']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            # mask class probabilities 
            probs =  F.softmax(output, dim=1).data.cpu().numpy()
            for zxy, prob in zip(tile_zxy, probs):
                foreground = prob[1:,:,:]
                reference =  np.linspace(0, 1, 256)
                foreground = np.digitize(foreground, reference).astype(np.uint8)

                palette = continuous_palette_for_color("pink", 256)
                out = Image.fromarray(foreground.squeeze(), mode="P")
                out.putpalette(palette)

                os.makedirs(os.path.join(result_path, 'probs'), exist_ok=True)
                path = os.path.join(result_path, 'probs', zxy + ".png")

                out.save(path, optimize=True)

            # Compute accuracy metrics
            pred = output.data.cpu().numpy() 
            target_cpu = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target_cpu[:,0,:,:], pred)

        
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        F1Score = self.evaluator.F1_Score()
        Recall = self.evaluator.Recall()
        Precision = self.evaluator.Precision()
        metrics_results = "Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, F1Score: {}, Recall: {}, Precision: {}".format(Acc, Acc_class, mIoU, FWIoU, F1Score, Recall, Precision)
        print('Test:')
        print(metrics_results)

        
        results = open(os.path.join(result_path,'Test_result_Boston.csv'), 'a', encoding='utf-8', newline='')
        f = csv.writer(results)
        f.writerow([metrics_results])
        results.close()





def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--best_model', type=str, default=None,
                        help='define the best model path')

    # misc
    parser.add_argument('--test_path', type=str, default='../../datasets/test/')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    print(args)
    torch.manual_seed(args.seed)
    predict = Predict(args)
    predict.predict()


if __name__ == "__main__":
   main()
