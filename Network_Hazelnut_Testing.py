#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
#from TestDataset import TestDataset
#from GroundtruthDataset import GroundtruthDataset
from HazelnutDataset import HazelnutDataset
from HazelnutDatasetGroundtruth import HazelnutDatasetGroundtruth
from torchvision import transforms
from Hazelnut_Scaling import Scaling
from ToTensor import ToTensor
from ToTensor_Groundtruth import ToTensor_Groudtruth
import numpy as np

#%%## Defining Network
class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(32,100, 8, stride=1, padding=0)
        
        self.convT1 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.convT4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.convT5 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.convT6 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.convT7 = nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1)
        self.convT8 = nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1)
        self.convT9 = nn.ConvTranspose2d(100, 32, 8, stride=1, padding=0)        
        
        self.bn0 = nn.BatchNorm2d(3)        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(100)

    def forward(self, x):
        
        # encoder
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        x = F.leaky_relu(x)
        x = self.conv7(x)
        x = F.leaky_relu(x)
        x = self.conv8(x)
        x = F.leaky_relu(x)
        x = self.conv9(x)
        x = F.sigmoid(x)
        
        # decoder
        x = self.convT9(x)
        x = F.sigmoid(x)
        x = self.convT8(x)
        x = F.leaky_relu(x)
        x = self.convT7(x)
        x = F.leaky_relu(x)
        x = self.convT6(x)
        x = F.leaky_relu(x)
        x = self.convT5(x)
        x = F.leaky_relu(x)
        x = self.convT4(x)
        x = F.leaky_relu(x)
        x = self.convT3(x)
        x = F.leaky_relu(x)
        x = self.convT2(x)
        x = F.leaky_relu(x)
        x = self.convT1(x)
        x = F.leaky_relu(x)
        
        return x

#%% Making Binary
def making_binary(img, device):
    r, g, b = img[:,0,:,:], img[:,1,:,:], img[:,2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    thresh = gray.mean()
    gray = torch.Tensor.cpu(gray).numpy()
    ret,thresh_img = cv2.threshold(gray,thresh,255,cv2.THRESH_TRUNC)
    thresh = thresh_img.mean()
    ret,thresh_img = cv2.threshold(thresh_img,thresh,255,cv2.THRESH_BINARY_INV)
    thresh_img = torch.tensor(thresh_img, device=device).float()
    return thresh_img

#%% Comparison of groundtruth and output
def issame(gt, ddd):
    return F.mse_loss(gt,ddd).float() #MSE

#%% Testing Step
def test(args, model, device, test_loader, groundtruth_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample1,sample2 in zip(test_loader, groundtruth_loader):
            
            data = sample1['image']
            data = data.to(device)
            data = data.float()
            
            plt.Figure()
            plt.imshow(torch.Tensor.cpu(data[0,:,:,:]).numpy().transpose(1,2,0).astype(int))
            plt.show()
            
            groundtruth = sample2['image']
            groundtruth = groundtruth.to(device)
            groundtruth = groundtruth.float()
            
#            plt.Figure()
#            plt.imshow(torch.Tensor.cpu(groundtruth[0,:,:]).numpy())
#            plt.show()
            
            output = model(data)
            
            plt.Figure()
            plt.imshow(torch.Tensor.cpu(output[0,:,:,:]).numpy().transpose(1,2,0).astype(int))
            plt.show()
            
            difference = data - output
            
#            plt.Figure()
#            plt.imshow(torch.Tensor.cpu(difference[0,:,:,:]).numpy().transpose(1,2,0))
#            plt.show()
            
            binary_difference = making_binary(difference, device)
            
            plt.Figure()
            plt.imshow(torch.Tensor.cpu(binary_difference[0,:,:]).numpy())
            plt.show()
            
            pred = issame(groundtruth,binary_difference)
            print(pred)
            correct += pred
    return correct

#%% Main
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Hazelnut')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    
    # https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

    # test dataset
#    test_dataset = TestDataset.dataset_concat_test()
    test_dataset = HazelnutDataset(r'test\cut',
                                   transform=transforms \
                                   .Compose([Scaling(), ToTensor()]))
    
#    groundtruth dataset
#    groundtruth_dataset = GroundtruthDataset.dataset_concat_groundtruth()
    groundtruth_dataset = HazelnutDatasetGroundtruth(r'ground_truth\cut',
                                                     transform=transforms \
                                                     .Compose([Scaling(), ToTensor_Groudtruth()]))
    
    # test data loader
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
     # groundtruth data loader
    groundtruth_loader = torch.utils.data.DataLoader(groundtruth_dataset,
                                              batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # make a network instance
    PATH = 'encoder_hazelnut_256.pt'
    model = Net().to(device)
    model.load_state_dict(torch.load(PATH))

    # train epochs
    correct = test(args, model, device, test_loader, groundtruth_loader)
    print(correct/len(groundtruth_loader))


    # run inference
#    if args.save_model:
#	    model = Net()
#	    model.load_state_dict(torch.load("lsp_cnn.pt"))
#	    sample = next(iter(valid_loader))
#	    model.eval()
#	    outputs = model(sample['image'].float())
#	    _, lbl = torch.max(outputs.data, 1)
#	    print('\nThe true lable: ', sample['landmarks'][0].item())
#	    print('The classifier lable: ', lbl[0].item())

#%% Calling main
if __name__ == '__main__':
    main()
