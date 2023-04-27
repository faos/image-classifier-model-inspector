import torch
import torchvision
import timm


def get_pytorch_models(name, num_categories, pretrained):
        if name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=pretrained)
            model.fc = torch.nn.Linear(512, num_categories)
        
        return model


class ModelLoader(object):

    def __init__(self, framework, model_name, num_categories, pretrained=False, weight_path=None):
        self.framework = framework
        self.model_name = model_name
        self.num_categories = num_categories
        self.pretrained = pretrained
        self.weight_path = weight_path
        self.model = None

        if framework == "timm":
            self.timm_loader()
        elif framework == "pytorch":
            self.pytorch_loader()

    def timm_loader(self):
        self.model = timm.create_model(
            self.model_name, 
            pretrained=self.pretrained, 
            num_classes=self.num_categories
            )

        if self.weight_path:
            pth = torch.load(self.weight_path)
            print(pth)
            if isinstance(pth, dict):
                if 'best_ckp' in pth:
                    self.model.load_state_dict(pth['best_ckp'])
                elif 'state_dict' in pth:
                    self.model.load_state_dict(pth['state_dict'])
                elif 'model-state-dict' in pth:
                    self.model.load_state_dict(pth['model-state-dict'])
            else:
                self.model.load_state_dict(pth)
        
        self.model.eval()

    def pytorch_loader(self):
        print('pytorch_loader'.upper())

        self.model = get_pytorch_models(
            self.model_name, 
            self.num_categories, 
            self.pretrained
            )
        
        device = torch.device('cpu')

        if self.weight_path:
            pth = torch.load(self.weight_path, map_location=device)
            print(pth)
            if isinstance(pth, dict):
                if 'best_ckp' in pth:
                    self.model.load_state_dict(pth['best_ckp'])
                elif 'state_dict' in pth:
                    self.model.load_state_dict(pth['state_dict'])
                elif 'model-state-dict' in pth:
                    self.model.load_state_dict(pth['model-state-dict'])
            else:
                self.model.load_state_dict(pth)
        
        self.model.eval()


if __name__ == '__main__':
    loader = ModelLoader('timm', 'resnet50', 5, False)
    loader = ModelLoader('timm', 'vgg19', 5, False)
    loader = ModelLoader('timm', 'densenet121', 5, False)