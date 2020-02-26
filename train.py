# code adapted from: https://github.com/jfzhang95/pytorch-deeplab-xception


import argparse
import os
import numpy as np
import pandas as pd 
from tqdm import tqdm

from modeling.dataloaders import building
#from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab.deeplab_model import *
from modeling.utils.loss import SegmentationLosses
from modeling.utils.calculate_weights import calculate_weigths_labels
from modeling.utils.lr_scheduler import LR_Scheduler
from modeling.utils.saver import Saver
from modeling.utils.summaries import TensorboardSummary
from modeling.utils.metrics import Evaluator

class Trainer(object):
    def __init__(self, args, gpu = None):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        expect_dist_w = {'wce','dicewce'}
        if args.loss_type in expect_dist_w:
            self.weighted_loss_function = 1
        else:
            self.weighted_loss_function = 0
        
        # Define Dataloader
        #kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader = building.get_loader(image_path=args.train_path,
                                image_size=512,
                                batch_size=args.batch_size,
                                num_workers=args.workers,
                                data_type='train',
                                augment_prob=args.augment_prob,
                                weighted_loss_function=self.weighted_loss_function,
                                sigma =args.sigma,
                                w0 =args.w0,
                                ddp = args.ddp,
                                ntrain_subset = args.ntrain_subset)#
        self.val_loader = building.get_loader(image_path=args.val_path,
                                image_size=512,
                                batch_size=args.batch_size,
                                num_workers=args.workers,
                                data_type='valid',
                                augment_prob=0.,
                                weighted_loss_function=0)#

        self.nclass = 2
        print("GPU is available")
        print(torch.cuda.is_available())
        print("Current GPU devices:")
        print(torch.cuda.current_device())
        print("GPU devices count:")
        print(torch.cuda.device_count())

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        if args.optim == 'sgd':
            print("SGD Optim")
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)
        if args.optim == 'Adam':
            print("Adam Optim")
            optimizer = torch.optim.Adam(list(model.parameters()),
                                         args.lr, [args.beta1, args.beta2])

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
        #                                     args.epochs, len(self.train_loader))
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

        # Using cuda
        if args.cuda:
            if len(args.gpu_ids) > 1 :
                self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
                #patch_replication_callback(self.model)
                self.model = self.model.cuda()
            else:
                print("Just one GPU")
                self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        self.acc_log = pd.DataFrame(columns=['Epoch', 'Acc', 'Acc_class', 'mIoU', 'FwIoU', 'F1Score', 'Recall', 'Precision'])
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if self.args.cuda and len(self.args.gpu_ids) > 1:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.acc_log = checkpoint['acc_log']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        
        for i, sample in enumerate(tbar):
            if (self.weighted_loss_function == 1):
                image, target, dist_w = sample['image'], sample['label'], sample['dist_w']
                if self.args.cuda:
                    image, target, dist_w = image.cuda(), target.cuda(), dist_w.cuda()
            else:
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
            #self.scheduler(self.optimizer, i, epoch, self.best_pred)
            
            self.optimizer.zero_grad()
            output = self.model(image)
            
            if (self.weighted_loss_function == 1):
                loss = self.criterion(output, target,dist_w)
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if self.args.save_image:
                if i % (num_img_tr // 10) == 0:
                    global_step = i + num_img_tr * epoch
                    self.summary.visualize_image('train', self.writer, self.args.dataset, image, target, output, global_step)
            
        
        self.scheduler.step()
        print('\nLearning rate at this epoch is: %0.9f' % self.scheduler.get_lr()[0])

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        num_img_vl = len(self.val_loader)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            #loss = self.criterion(output, target[:,0,:,:])
            #test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            # Show 10 * 3 inference results each epoch
            if self.args.save_image:
                if i % (num_img_vl // 10) == 0:
                    global_step = i + num_img_vl * epoch
                    self.summary.visualize_image('val',self.writer, self.args.dataset, image, target, output, global_step)
            
            pred = output.data.cpu().numpy()
            
            target_cpu = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            
            # Add batch sample into evaluator
            self.evaluator.add_batch(target_cpu[:,0,:,:], pred)

        
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        F1Score = self.evaluator.F1_Score()
        Recall = self.evaluator.Recall()
        Precision = self.evaluator.Precision()
        acc_list = [epoch+1, Acc, Acc_class, mIoU, FWIoU, F1Score, Recall, Precision] 

        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/F1Score', F1Score, epoch)
        self.writer.add_scalar('val/Recall', Recall, epoch)
        self.writer.add_scalar('val/Precision', Precision, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, F1Score: {}, Recall: {}, Precision: {}".format(Acc, Acc_class, mIoU, FWIoU, F1Score, Recall, Precision))
        print('Loss: %.3f' % test_loss)

        self.acc_log.loc[epoch] = acc_list


        is_best = False

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if self.args.cuda and len(self.args.gpu_ids) > 1:
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(), # if using multiple gpu: self.model.module.state_dict()
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'acc_log': self.acc_log,
            }, is_best,)
        else:
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(), 
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'acc_log': self.acc_log,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='building',
                        help='dataset name (default: building)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', action='store_true', default=False,
                        help='whether to use sync bn (default: False)')
    parser.add_argument('--freeze-bn', action='store_true', default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal','wce','dice','dicece','dicewce'],
                        help='loss function type (default: ce)')
    parser.add_argument('--w0', type=int, default=10,
                        help='Unet loss function parameter w0')
    parser.add_argument('--sigma', type=int, default=5,
                        help='Unet loss function parameter sigma')
    parser.add_argument('--augment_prob', type=float, default=0.0,
                        help='data augmentation rate')
    parser.add_argument('--save_image', action='store_true', default=False,
                        help='wheter to save results images during training and validation or not (default: False)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    
    # optimizer params
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'Adam'],
                        help='optimizer name (default: sgd)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help= 'momentum1 in Adam') 
    parser.add_argument('--beta2', type=float, default=0.999,
                        help= 'momentum1 in Adam')     
    
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='wheter to use DistributedDataParallel or not (default: False)')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of HPC nodes')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    # results saving
    parser.add_argument('--results_path', type=str, default='results',
                        help='set the path of the folder where the results will be saved')
    parser.add_argument('--experiment', type=str, default=None,
                        help='set the experiment name')        

    # subset the training data at each checkpoint call useful if there is a time limit
    # that does not allow to finalize an epoch
    parser.add_argument('--ntrain_subset', type=int, default=None,
                        help='creat a random subset of ntrain_subset elements')                

    # checking point resume
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # Datasets
    parser.add_argument('--train_path', type=str, default='../../datasets/train/')
    parser.add_argument('--val_path', type=str, default='../../datasets/val/')
    parser.add_argument('--test_path', type=str, default='../../datasets/test/')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # if args.sync_bn is None:
    #     if args.cuda and len(args.gpu_ids) > 1:
    #         args.sync_bn = True
    #     else:
    #         args.sync_bn = False

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    # enable cudnn auto-tuner

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enable = True
    
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
