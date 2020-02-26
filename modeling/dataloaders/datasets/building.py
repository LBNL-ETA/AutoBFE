import os
import random
from random import shuffle
from random import sample
import numpy as np
import torch
from skimage.measure import label
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from dataprep.distance import unet_weight_map
from PIL import Image


class BldgDataLoad(data.Dataset):
    def __init__(self, root_dir,image_size=512, data_type='train',
                 augment_prob=0.4, weighted_loss_function = 0,
                 w0 = 10, sigma = 5, wc = {0: 1, 1: 1}, ntrain_subset = None):
        
        self.root_dir = root_dir
        self.image_size = image_size
        self.data_type = data_type
        self.augment_prob = augment_prob
        self.weighted_loss_function = weighted_loss_function
        self.RotationDegree = [0,90,180,270]
        self.w0 = w0
        self.wc = wc
        self.sigma = sigma
        self.image_paths = list(map(lambda x: os.path.join(root_dir, 'images', x),
                                os.listdir((os.path.join(root_dir, 'images')))))

        print("images count in {} path :{}".format(self.data_type,len(self.image_paths)))

        if ntrain_subset is not None:
            self.image_paths = sample(self.image_paths, ntrain_subset)
            print("trainng subset images count in {} path :{}".format(self.data_type,len(self.image_paths)))

        # self.image_paths =[os.path.join(root_dir, 'images', f) 
        #                    for f in os.listdir(os.path.join(root_dir, 'images')) 
        #                    if os.path.isfile(os.path.join(root_dir,'images', f))]


    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('/images/')[-1][:-len(".jpg")]
        filename_mask = filename + '.png'
        mask_path = os.path.join(self.root_dir, 'masks', filename_mask)
        
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)#.convert('L')
        # convert RGB into B & W
        # TODO: Define a more general rules
        #thresh = np.asarray(mask).mean()+ 10
        
        thresh = 100
        fn = lambda x : 255 if x < thresh else 0
        mask = mask.convert('L').point(fn, mode='1')
       

        #mask = mask.convert('L').point(rgb_to_bw, mode='1')
        
        if(self.data_type == 'train') and (self.weighted_loss_function == 1):
            filename_dist = filename + '.npy'
            dist_path = os.path.join(self.root_dir, 'distances', filename_dist)
            dist = np.load(dist_path)
            dist_w = unet_weight_map(mask, dist, wc= self.wc, w0 = self.w0, sigma = self.sigma)

        
        Transform = []

        p_trans = random.random()
        if (self.data_type == 'train') and (p_trans < self.augment_prob):

            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
                if(self.data_type == 'train') and (self.weighted_loss_function == 1):
                    dist_w = np.flip(dist_w,axis=1)

            if random.random() < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)
                if(self.data_type == 'train') and (self.weighted_loss_function == 1):
                    dist_w = np.flip(dist_w,axis=0)

            Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

            image = Transform(image)

            Transform =[]

        
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        
        image = Transform(image)
        mask = Transform(mask)
        #Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        Norm_ = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = Norm_(image)
        
        if(self.data_type == 'train') and (self.weighted_loss_function == 1):
            dist_w = np.ascontiguousarray(dist_w)
            dist_w = torch.from_numpy(dist_w)
            sample = {'image': image, 'label': mask, 'dist_w': dist_w, 'tile_zxy':filename}
        else:
            sample = {'image': image, 'label': mask, 'tile_zxy':filename}
        return sample

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)
    
def rgb_to_bw(x):
    return 255 if x > 100 else 0

def get_loader(image_path, image_size, batch_size, num_workers= 2 ,
               data_type = 'train', augment_prob=0.4,
               weighted_loss_function=0, w0 = 10, sigma = 5, 
               ddp = False, world_size = None, rank = None, ntrain_subset = None):
    """Builds and returns Dataloader."""

    dataset = BldgDataLoad(root_dir = image_path, image_size =image_size, 
                           data_type = data_type, augment_prob= augment_prob,
                           weighted_loss_function=weighted_loss_function,w0 = w0, 
                           sigma = sigma, ntrain_subset = ntrain_subset)

    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      sampler = train_sampler)
    elif data_type == 'train':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      drop_last=True)

    
    return data_loader