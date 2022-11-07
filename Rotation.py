import imgaug.augmenters as iaa
from random import randrange

class Rotation(object):
    
    def __init__(self, show = []):
        self.show = show
        
    def __call__(self, sample):
        image = sample
        StochasticParameter = randrange(-360,360)
        seq = iaa.Sequential([iaa.geometric.Rotate(rotate = StochasticParameter)])
        img_aug = seq(image=image)
        
        return img_aug
