import os
import json
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch

class LeafDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, split_file, 
                 image_set='train', img_transform=None, mask_transform=None):
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.image_set = image_set
        
        # Read split file to get image names
        with open(split_file, 'r') as f:
            file_names = [x.strip() for x in f.readlines()]
        
        # Construct full paths
        self.images = [os.path.join(image_dir, fname + '.jpg') for fname in file_names]
        self.masks = [os.path.join(mask_dir, fname + '_binarymask.jpg') for fname in file_names]
        
        # Verify that all files exist
        self._verify_files()
        
    def _verify_files(self):
        """Verify that all files exist"""
        for img_path, mask_path in zip(self.images, self.masks):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        mask_array = np.array(mask)
        mask_array = (mask_array > 128).astype(np.uint8) 
        mask_array = mask_array * 255
        # Convert to grayscale
        mask = Image.fromarray(mask_array.astype(np.uint8))

        if self.img_transform is not None:
            img = self.img_transform(img)
        
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.images)

    @staticmethod
    def decode_target(mask):
        """Decode binary mask to RGB image"""
        leaf_color = [255, 255, 255]
        
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[mask == 1] = leaf_color

        return Image.fromarray(rgb_mask)
