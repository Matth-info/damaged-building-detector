# Dataset folder keep track of the custom pytorch dataset that have been used to load, preprocess data according to the source dataset and the model specificity
import os 
import torch 
import albumentation
import numpy as np
import pandas as pd 
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

class Cloud_38_Dataset(Dataset):
    """ 
    This Dataset allows to load the 38-Cloud Dataset
    The images are loaded by concatenating 3 or 4 input channels

    base_path = Path('/home/onyxia/work/38-Cloud_training')
    data = CloudDataset(base_path/'train_red', 
                    base_path/'train_green', 
                    base_path/'train_blue', 
                    base_path/'train_nir',
                    base_path/'train_gt')
    """
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True, include_nir=False, transform=None):
        super().__init__()
        
        # Loop through the files in the red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        self.include_nir = False
        self.transform = transform
        
    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        # Combine file paths for different spectral bands into a dictionary
        files = {'red': r_file, 
                 'green': g_dir / r_file.name.replace('red', 'green'),
                 'blue': b_dir / r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir / r_file.name.replace('red', 'nir'),
                 'gt': gt_dir / r_file.name.replace('red', 'gt')}
        return files
                                       
    def __len__(self):
        # Return the number of files in the dataset
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        # Open image files as arrays, optionally including NIR channel
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                           ], axis=2)
    
        if self.include_nir: # Include Near Infrared channel
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
    
        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))
    
        # Normalize pixel values
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    
    def open_mask(self, idx, add_dims=False):
        # Open ground truth mask as an array
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask == 255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        # Get an item from the dataset (image and mask)
        x = self.open_as_array(idx, invert=self.pytorch)
        y = self.open_mask(idx, add_dims=False)

        if self.transform: # apply transformation to x # data augmentation
            augmented = self.transform(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']
            
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)

        
        return x, y
    
    def open_as_pil(self, idx):
        # Open an image as a PIL image
        arr = 256 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        # Return a string representation of the dataset
        s = 'Dataset class with {} files'.format(self.__len__())
        return s
    
    def display_data(self, list_indices : List[int]):
        # Create a subplot grid for displaying the images and masks
        fig, ax = plt.subplots(5, 2, figsize=(15, 15))

        for i, idx in enumerate(list_indices):
            ax[i, 0].imshow(self.open_as_array(idx))
            ax[i, 0].set_title(f'Sample {i + 1}')
            
            ax[i, 1].imshow(self.open_mask(idx))
            ax[i, 1].set_title(f'Sample {i + 1}: Ground truth')

        plt.tight_layout()
        plt.show()
