import torchvision.transforms as T
from timm.data.transforms_factory import create_transform
import torch.nn.functional as F 
import torch


class Predictor():

    def __init__(self, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, model, x):
        model.eval()
        transformer = create_transform(
            input_size=self.size, 
            mean=self.mean, 
            std=self.std
            )
        img = transformer(x).unsqueeze(0)    
        logits = model(img)
        probabilities = F.softmax(logits).flatten().cpu().tolist()

        return probabilities


class BatchPredictorFromTensor():

    def __init__(self, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, model, x):
        model.eval()
        print("BatchPredictorFromTensor", x.shape)
        img = T.Normalize(self.mean, self.std)(x)

        logits = model(img)
        print('\tLogits:')
        print(logits.detach().cpu().numpy().round(3))

        probabilities = F.softmax(logits, dim=1).detach().cpu()
        
        values, indexes = torch.max(probabilities, dim=1)

        prob_np = probabilities.numpy()
        print('\tProbabilities:')
        print(prob_np)

        return prob_np