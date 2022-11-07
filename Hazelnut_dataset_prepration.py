#%% Importing Librarires
from __future__ import print_function, division
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from HazelnutDataset import HazelnutDataset
from ToTensor import ToTensor
from Hazelnut_Scaling import Scaling
import numpy as np
from HazelnutDatasetGroundtruth import HazelnutDatasetGroundtruth
from ToTensor_Groundtruth import ToTensor_Groudtruth

#%% Ground Truth
#transformed_dataset = HazelnutDatasetGroundtruth(r'ground_truth\crack',
#                                      transform=transforms \
#                                      .Compose([Scaling(), ToTensor_Groudtruth()]))
#test_image = transformed_dataset[13]
#plt.figure()
#plt.imshow(test_image['image'])

#%% Train dataset
transformed_dataset = HazelnutDataset(r'train\good',
                                      transform=transforms \
                                      .Compose([Scaling(), transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
test_image = transformed_dataset[50]
test_image = test_image.numpy().transpose(1,2,0)
plt.figure()
#plt.imshow(test_image['image'])
plt.imshow(test_image)
plt.show()
#img = test_image['image']
img = test_image

#def making_binary(img):
#    rgb = 1
#    thresh1 = np.mean((max(img[:,0,rgb]), min(img[:,0,rgb])))
#    thresh2 = np.mean((max(img[:,1,rgb]), min(img[:,1,rgb])))
#    thresh3 = np.mean((max(img[:,2,rgb]), min(img[:,2,rgb])))
#    threshold = np.mean((thresh1, thresh2, thresh3))
#    img_thresh = np.zeros([len(img), len(img)])
#    img_thresh[img[:,:,rgb]<=threshold] = 0
#    img_thresh[img[:,:,rgb]>threshold] = 255
#    threshold = np.mean((np.ndarray.max(img), np.ndarray.min(img)))
#    img_thresh = np.zeros([len(img), len(img),3])
#    img_thresh[img<=threshold] = 0
#    img_thresh[img>threshold] = 255
#    return img_thresh

#plt.figure()
#plt.imshow(making_binary(img))
#plt.show()
    
def rgb2gray(img):

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

gray = rgb2gray(img)
plt.figure()
plt.imshow(gray)

thresh = np.mean(gray)
ret,thresh_img = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)

plt.figure()
plt.imshow(thresh_img)
plt.show()


    
#%% Test dataset --> good --> 0-39
test_dataset_good = HazelnutDataset(r'test\good',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))

test_image = test_dataset_good[18]
plt.imshow(test_image['image'])
    
#%% Test dataset --> crack --> 0-17
test_dataset_crack = HazelnutDataset(r'test\crack',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))

test_image = test_dataset_crack[10]
plt.imshow(test_image['image'])
    
#%% Test dataset --> cut --> 0-16
test_dataset_cut = HazelnutDataset(r'test\cut',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
test_image = test_dataset_cut[5]
plt.imshow(test_image['image'])
    
#%% Test dataset --> hole --> 0-17
test_dataset_hole = HazelnutDataset(r'test\hole',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
test_image = test_dataset_hole[12]
plt.imshow(test_image['image'])

#%% Test dataset --> print --> 016
test_dataset_print = HazelnutDataset(r'test\print',
                                      transform=transforms \
                                      .Compose([Scaling(), ToTensor()]))
test_image = test_dataset_print[6]
plt.imshow(test_image['image'])
