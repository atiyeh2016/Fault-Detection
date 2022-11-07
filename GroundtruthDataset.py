from HazelnutDatasetGroundtruth import HazelnutDatasetGroundtruth
from ToTensor import ToTensor
from Hazelnut_Scaling import Scaling
from torchvision import transforms
import torch

class GroundtruthDataset:
    
    def dataset_concat_groundtruth():
        good = HazelnutDatasetGroundtruth(r'ground_truth\good',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
        
        crack = HazelnutDatasetGroundtruth(r'ground_truth\crack',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
        cut = HazelnutDatasetGroundtruth(r'ground_truth\cut',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
        hole = HazelnutDatasetGroundtruth(r'ground_truth\hole',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
        printt = HazelnutDatasetGroundtruth(r'ground_truth\print',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))

        return torch.utils.data.ConcatDataset((good, crack, cut, hole, printt))
