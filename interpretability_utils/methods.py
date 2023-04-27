from captum.attr import Deconvolution, Saliency, LayerGradCam, LayerAttribution, GuidedBackprop, GuidedGradCam
from captum.attr import IntegratedGradients, InputXGradient, DeepLift, DeepLiftShap, GradientShap, ShapleyValueSampling, Deconvolution, Occlusion, IntegratedGradients

import torchvision.transforms as T
from timm.data.transforms_factory import create_transform
import torch.nn.functional as F 


class Interpreter():

    def __init__(self, model, method, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.model = model
        self.method = method
        self.size = size
        self.mean = mean
        self.std = std

        for m in self.model.modules():
            #print(m,)
            if hasattr(m, 'inplace'):
                m.inplace = False

    def get_method(self):
        if self.method == "saliency":
            return Saliency(self.model)
        elif self.method == 'input-x-gradient':
            return InputXGradient(self.model)
        elif self.method == 'integratedgradients':
            return IntegratedGradients(self.model)
        elif self.method == 'deconvolution':
            return Deconvolution(self.model)
        elif self.method == 'guided-backpropagation':
            return GuidedBackprop(self.model)
        elif self.method == 'DeepLift':
            return DeepLift(self.model)
            
    def __call__(self, x, y):
        attribution = self.get_method()

        transformer = create_transform(
            input_size=self.size, 
            mean=self.mean, 
            std=self.std
            )
        img = transformer(x).unsqueeze(0)

        att_map = attribution.attribute(img, target=y).detach().cpu().numpy()[0]

        att_map = self.prepare_result(att_map)

        return att_map

    def prepare_result(self, matrix):
        if len(matrix.shape) == 3:
            matrix = matrix.mean(0)

        mmin = matrix.min()
        mmax = matrix.max()

        matrix = (matrix - mmin)/(mmax - mmin)
        
        return matrix
