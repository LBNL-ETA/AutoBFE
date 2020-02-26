import os
import shutil
import torch
import torchvision
from collections import OrderedDict
import glob
import re
import pandas as pd

# Functions to sort experiments files based on the experiments numbers
# From https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

# def atoi(text):
#     return int(text) if text.isdigit() else text

# def natural_keys(text):
#     return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.experiment_dir = os.path.join(args.results_path, args.dataset, args.experiment)
        self.examples_dir = os.path.join(self.experiment_dir, 'examples')
        # self.directory = os.path.join(args.results_path, args.dataset, args.experiment)
        # #self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        # runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
        # runs.sort(key=natural_keys)
        # self.runs = runs
        # self.run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        # self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(self.run_id )))
        # self.examples_dir = os.path.join(self.directory, 'experiment_{}'.format(str(self.run_id )),'examples')

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        if not os.path.exists(self.examples_dir):
            os.makedirs(self.examples_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        acc_log = state['acc_log']
        acc_log.to_csv(os.path.join(self.experiment_dir, 'accuracy_log.txt'),index=False)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'best_model.pth.tar'))
            
    def save_examples(self, image, pred, target, epoch):
        torchvision.utils.save_image(image.data.cpu(),
                                    os.path.join(self.examples_dir,'valid_%d_image.png'%(epoch+1)))
        torchvision.utils.save_image(pred.data.cpu(),
                                    os.path.join(self.examples_dir,'valid_%d_Prediction.png'%(epoch+1)))
        torchvision.utils.save_image(target.data.cpu(),
                                    os.path.join(self.examples_dir,'valid_%d_GT.png'%(epoch+1)))



    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = vars(self.args)
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()