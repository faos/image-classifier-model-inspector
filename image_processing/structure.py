import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random


class PatchTransform(object):
    def __init__(self, k = 2):
        self.k = k

    def __call__(self, xtensor:torch.Tensor):
        '''
        X: torch.Tensor of shape(c, h, w)   h % self.k == 0
        :param xtensor:
        :return:
        '''
        patches = []
        c, h, w = xtensor.size()
        dh = h // self.k
        dw = w // self.k

        #print(dh, dw)
        sh = 0

        for i in range(h // dh):
            eh = sh + dh
            eh = min(eh, h)
            sw = 0
            for j in range(w // dw):
                ew = sw + dw
                ew = min(ew, w)
                patches.append(xtensor[:, sh:eh, sw:ew])

                #print(sh, eh, sw, ew)
                sw = ew
            sh = eh

        random.shuffle(patches)

        start = 0
        imgs = []

        for i in range(self.k):
            end = start + self.k
            imgs.append(torch.cat(patches[start:end], dim = 1))
            start = end

        img = torch.cat(imgs, dim = 2)

        return img


class HorizontalTransform(object):
    def __init__(self, k = 2):
        self.k = k

    def __call__(self, xtensor:torch.Tensor):
        '''
        X: torch.Tensor of shape(c, h, w)   h % self.k == 0
        :param xtensor:
        :return:
        '''
        patches = []
        c, h, w = xtensor.size()
        dh = h // self.k
        
        sh = 0

        for i in range(h // dh):
            eh = sh + dh
            eh = min(eh, h)
                        
            sub = xtensor[:, sh:eh, :]
            patches.append(sub)
            
            print('SUB:', sub.shape)

            sh = eh

        random.shuffle(patches)

        start = 0
        imgs = []

        #for i in range(self.k):
        #    end = start + self.k
        #    imgs.append(torch.cat(patches[start:end], dim = 1))
        #    start = end

        img = torch.cat(patches, dim=1)

        return img


class VerticalTransform(object):
    def __init__(self, k = 2):
        self.k = k

    def __call__(self, xtensor:torch.Tensor):
        '''
        X: torch.Tensor of shape(c, h, w)   h % self.k == 0
        :param xtensor:
        :return:
        '''
        patches = []
        c, h, w = xtensor.size()
        dh = w // self.k
        
        sh = 0

        for i in range(h // dh):
            eh = sh + dh
            eh = min(eh, h)
                        
            sub = xtensor[:, :, sh:eh]
            patches.append(sub)
            
            print('SUB:', sub.shape)

            sh = eh

        random.shuffle(patches)

        start = 0
        imgs = []

        #for i in range(self.k):
        #    end = start + self.k
        #    imgs.append(torch.cat(patches[start:end], dim = 1))
        #    start = end

        img = torch.cat(patches, dim=2)

        return img


class PatchShuffle():
    parameters = {"patch_size":True}
    
    def __init__(self, patch_size, **kwargs):
        self.patch_size = patch_size
        self.transformer = PatchTransform(self.patch_size)
        self.resize = transforms.Resize((224, 224))
    
    def __call__(self, x):
        print('PatchShuffle:')
        print('IMG SIZE:\t', x.size)
        transform_raw_size = transforms.Resize(x.size)

        x1 = transforms.ToTensor()(self.resize(x))

        print(type(x1))

        ans = self.transformer(x1).permute(1, 2, 0).numpy() * 255



        print("Max and Min: \t", ans.max(), ans.min())

        return ans


class HorizontalShuffle():
    parameters = {"patch_size":True}

    def __init__(self, patch_size, **kwargs):
        self.patch_size = patch_size
        self.transformer = HorizontalTransform(self.patch_size)
        self.resize = transforms.Resize((224, 224))
    
    def __call__(self, x):
        print('PatchShuffle:')
        print('IMG SIZE:\t', x.size)
        transform_raw_size = transforms.Resize(x.size)

        x1 = transforms.ToTensor()(self.resize(x))

        print(type(x1))

        ans = self.transformer(x1).permute(1, 2, 0).numpy() * 255



        print("Max and Min: \t", ans.max(), ans.min())

        return ans


class VerticalShuffle():
    parameters = {"patch_size":True}

    def __init__(self, patch_size, **kwargs):
        self.patch_size = patch_size
        self.transformer = VerticalTransform(self.patch_size)
        self.resize = transforms.Resize((224, 224))
    
    def __call__(self, x):
        print('PatchShuffle:')
        print('IMG SIZE:\t', x.size)
        transform_raw_size = transforms.Resize(x.size)

        x1 = transforms.ToTensor()(self.resize(x))

        print(type(x1))

        ans = self.transformer(x1).permute(1, 2, 0).numpy() * 255



        print("Max and Min: \t", ans.max(), ans.min())

        return ans