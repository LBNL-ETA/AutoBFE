import os
import sys
import concurrent.futures as futures
import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from PIL import Image


def distances_map(mask):

    """
    From: https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    
    labels = label(mask)
    # no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        distances = distances[:,:,0:2]
        distances = np.sum(distances, axis=2)
    else:
        distances = np.zeros_like(mask)
    
    return distances.astype(np.float16)

def distance_weight(root_dir,num_workers=4):
    """Compute distance weight for each mask.

    Args:
      url_map: to fetch satellite images from
      image_format: used image fomat
      tiles_cover_path: path to csv file to store tiles in
      sat_path: path to slippy map directory for storing satellite images tiles
      num_workers: rate limit in max. requests per second

    """
    image_paths = list(map(lambda x: os.path.join(root_dir, 'images', x),
                       os.listdir((os.path.join(root_dir, 'images')))))

    # Create directories if not exist
    distances_path = os.path.join(root_dir, 'distances')
    if not os.path.exists(distances_path):
        os.makedirs(distances_path)

    with futures.ThreadPoolExecutor(num_workers) as executor:

        def worker(image_path):

            filename = image_path.split('/images/')[-1][:-len(".jpg")]
            filename_mask = filename + '.png'
            filename_w = filename + '.npy'
            mask_path = os.path.join(root_dir, 'masks', filename_mask)
            w_path = os.path.join(root_dir, 'distances', filename_w)
            mask = Image.open(mask_path)
            thresh = 100
            fn = lambda x : 255 if x > thresh else 0
            mask = mask.convert('L').point(fn, mode='1')
            mask = np.array(mask)
            w = distances_map(mask)
            np.save(w_path, w)
            return image_path, True

        for image_path, ok in executor.map(worker, image_paths):
            if not ok:
                print("Warning: {} does not exist".format(image_path), file=sys.stderr)

def unet_weight_map(mask, distance, wc=None, w0 = 10, sigma = 5):

    """
    From: https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    distance: Numpy array
        2D array of shape (image_height, image_width) representing distance map
        of mask.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    mask = np.array(mask)
    labels = label(mask)
    no_labels = labels == 0

    distance = distance.astype(np.float64)

    weights0 = w0 * np.exp(-1/2*((distance) / sigma)**2) * no_labels

    if wc:
        mask = 1.* mask
        class_weights = np.zeros_like(mask)
        for k, v in wc.items():
            class_weights[mask == k] = v
        weights = weights0 + class_weights
    
    return weights