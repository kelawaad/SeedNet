import scipy
import cv2 as cv
import numpy as np
from Utils import *
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from skimage import measure
from skimage import data, filters
from skimage.segmentation import random_walker
# import matplotlib.pyplot as plt



Class SeedNetEnv():
    def __init__(self, path='./MSRA_Dataset/MSRA10K_Imgs_GT/Imgs/', num_initial_seeds=3):
        self.path = Path(path)
        self.images, self.masks = load_data(self.path)
        self.images, self.masks = resize_data(iself.images, self.masks)
        self.images = np.array(self.images)
        self.masks  = np.array(self.masks)[:,:,:,0]
        self.masks[self.masks>0] = 255
        self.img = None
        self.all_masks_with_regions = get_all_different_regions(masks, kernel_size=(5,5))
        self.initial_foreground_seeds, self.initial_background_seeds = generate_initial_seeds(self.all_masks_with_regions)
        print(self.images.shape, self.masks.shape)
        print(f'\nEnvironment initialized.\nThe data contains {len(self.images)} images')    


    def test_env(self):
        fig, axs = plt.subplots(1, 3, figsize=(15,15))
        idx = 6262
        axs[0].imshow(images[idx])
        axs[1].imshow(masks[idx], cmap='gray')
        axs[2].imshow(all_masks_with_regions[idx], cmap='hot')
        axs[2].scatter(initial_foreground_seeds[idx, :, 1], initial_foreground_seeds[idx, :, 0], c='g')
        axs[2].scatter(initial_background_seeds[idx, :, 1], initial_background_seeds[idx, :, 0], c='r')
    
    def seed(self, seed=1337):
        np.random.seed(seed)
        print(f'Numpy seed with {seed}')
    
    def step(self, action):
        """
        use given action (seed) to generate new mask (Segmentation)

        """
        pass

    def reset(self, img_id):
        self.img = self.images[img_id]
        """
        takes img_id resetting the history to the img and segmentaion mask using the intial random seeds
        """
        pass

