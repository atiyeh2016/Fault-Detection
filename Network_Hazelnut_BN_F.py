#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from HazelnutDataset import HazelnutDataset
from torchvision import transforms
from Hazelnut_Scaling import Scaling
from ToTensor import ToTensor
from ToTensor_Groundtruth import ToTensor_Groudtruth
import numpy as np
import torch.optim as optim
from HazelnutDatasetGroundtruth import HazelnutDatasetGroundtruth
import copy
from torch.optim.lr_scheduler import StepLR
import time

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
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.leaky_relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.sigmoid(x)
        x = self.conv8(x)
#        x = self.bn8(x)
#        x = F.sigmoid(x)
#        x = self.conv9(x)
                
        # decoder
        x = self.convT8(x)
        x = F.sigmoid(x)
        x = self.bn7(x)
        x = self.convT7(x)
        x = self.bn6(x)
        x = F.leaky_relu(x)
        x = self.convT6(x)
        x = self.bn5(x)
        x = F.leaky_relu(x)
        x = self.convT5(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.convT4(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.convT3(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.convT2(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.convT1(x)
#        x = self.bn1(x)
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

#%% Training Step
loss_trend = []
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        data = sample
        target = copy.deepcopy(sample)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.float()
        output = model(data)
        loss = F.mse_loss(output, target.view_as(output).float())
        loss.backward()
        optimizer.step()
        loss_trend.append(loss.item())
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                loss.item()))

#%%## Testing Step
def test(args, model, device, test_loader, groundtruth_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample1,sample2 in zip(test_loader, groundtruth_loader):
            data = sample1
            data = data.to(device)
            data = data.float()
            groundtruth = sample2
            groundtruth = groundtruth.to(device)
            groundtruth = groundtruth.float()
            output = model(data)
#            print(output)
            plt.Figure()
            plt.imshow(torch.Tensor.cpu(output[0,:,:,:]).numpy().transpose(1,2,0))
            plt.show()
            
#            difference = data - output
#            binary_difference = making_binary(difference, device)
#            pred = issame(groundtruth,binary_difference)
#            print(pred)
#            correct += pred

#%% Main
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LSP')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 100)')
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

    # train dataset
    train_set = HazelnutDataset(r'train\good',transform=transforms \
                                .Compose([Scaling(), transforms.ToTensor()]))#    # test dataset
                                
                                
#    test_set = PoseLandmarksDataset('joints_test.csv', r'_images',
#                                    transform=transforms.Compose([ToTensor()]))

    # train data loader
    train_loader = torch.utils.data.DataLoader(train_set, 
    	batch_size=args.batch_size, shuffle=True, **kwargs)

    # test data loader
    test_dataset = HazelnutDataset(r'test\cut',
                                   transform=transforms \
                                   .Compose([Scaling(),transforms.ToTensor()])                                       )
                                   
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    groundtruth_dataset = HazelnutDatasetGroundtruth(r'ground_truth\cut',
                                                     transform=transforms \
                                                     .Compose([Scaling(), transforms.ToTensor()]))
                                                     
    groundtruth_loader = torch.utils.data.DataLoader(groundtruth_dataset,
                                              batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # make a network instance
    model = Net().to(device)

    # configure optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # configure learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train epochs
    
    tot_loss = np.zeros([args.epochs+1,1])
    for epoch in range(1, 3 + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, groundtruth_loader)
#        scheduler.step()
#        if not epoch%20:
    time_end = time.time()

#            valid(args, model, device, train_loader, args.fraction)
#        scheduler.step()

    # save the trained model
    if args.save_model:
	    torch.save(model.state_dict(), "encoder_hazelnut_Aug_F_1_200.pt")

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
