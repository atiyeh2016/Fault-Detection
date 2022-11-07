#%% Importing 
from __future__ import print_function, division
import os
import torch
from skimage import io
from torch.utils.data import Dataset

#%% Class Defenition
class HazelnutDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        path, dirs, files = next(os.walk(self.root_dir))
        return len(files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, f'{idx:03d}.png')
        image = io.imread(img_name)

#        sample = {'image': image}
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample