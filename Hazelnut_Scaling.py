import imgaug.augmenters as iaa

class Scaling(object):
    
     def __call__(self, sample, scale=0.25):
        image = sample
#        image = sample['image']
        
        seq = iaa.Sequential([iaa.geometric.Affine(scale=scale),
                              iaa.CenterCropToFixedSize(256,256)])
        image_aug = seq(image=image)
        
#        return {'image': image_aug}
        return image_aug
    
    
    

        
        