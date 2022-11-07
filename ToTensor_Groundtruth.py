import torch
import numpy as np

class ToTensor_Groudtruth(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, idx = sample['image'], sample['idx']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.from_numpy(image),
                'idx': idx}