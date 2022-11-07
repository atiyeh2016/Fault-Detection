from HazelnutDataset import HazelnutDataset
from ToTensor import ToTensor
from Hazelnut_Scaling import Scaling
from torchvision import transforms
import torch

class TestDataset:
    
    def dataset_concat_test():
        good = HazelnutDataset(r'test\good',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
        
        crack = HazelnutDataset(r'test\crack',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
        cut = HazelnutDataset(r'test\cut',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
        hole = HazelnutDataset(r'test\hole',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
        printt = HazelnutDataset(r'test\print',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))

        return torch.utils.data.ConcatDataset((good, crack, cut, hole, printt))
