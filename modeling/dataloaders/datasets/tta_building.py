import os
import random
from random import shuffle
import numpy as np
import torch
from skimage.measure import label
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image



class TtaBldgDataLoad(data.Dataset):
    def __init__(self, root_dir,image_size=512, data_type='test'):
        
        self.root_dir = root_dir
        self.image_size = image_size
        self.data_type = data_type
        self.RotationDegree = [0,90,180,270]
        self.image_paths = list(map(lambda x: os.path.join(root_dir, 'images', x),
                                os.listdir((os.path.join(root_dir, 'images')))))

        print("images count in {} path :{}".format(self.data_type,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('/images/')[-1][:-len(".jpg")]
        filename_mask = filename + '.png'
        mask_path = os.path.join(self.root_dir, 'masks', filename_mask)
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)#.convert('L')
        # convert RGB into B & W
        thresh = 100
        fn = lambda x : 255 if x > thresh else 0
        mask = mask.convert('L').point(fn, mode='1')


        # image augmentation
        image_hflip = F.hflip(image)
        image_vflip = F.vflip(image)
        image_90 = F.rotate(image,90)
        image_180 = F.rotate(image,180)
        image_270 = F.rotate(image,270)
        
        Transform = []
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        image_hflip = Transform(image_hflip)
        image_vflip = Transform(image_vflip)
        image_90 = Transform(image_90)
        image_180 = Transform(image_180)
        image_270 = Transform(image_270)
        mask = Transform(mask)

        Norm_ = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = Norm_(image)
        image_hflip = Norm_(image_hflip)
        image_vflip = Norm_(image_vflip)
        image_90 = Norm_(image_90)
        image_180 = Norm_(image_180)
        image_270 = Norm_(image_270)

        sample = {'image': image, 'label': mask, 'tile_zxy':filename,
                  'image_hflip': image_hflip, 'image_vflip': image_vflip, 
                  'image_90': image_90, 'image_180': image_180, 'image_270': image_270}
       
        return sample

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers= 2 ,data_type = 'train'):
    """Builds and returns Dataloader."""
    dataset = TtaBldgDataLoad(root_dir = image_path, image_size =image_size, data_type = data_type)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle = False,
                                  num_workers=num_workers)
    return data_loader