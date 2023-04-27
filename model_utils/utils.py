import timm

def get_framework_options(framework_name):
    if framework_name == 'pytorch':
        return get_pytorch_options()
    elif framework_name == 'timm':
        return get_timm_options()
    else:
        return []


def get_pytorch_options():
    opt = ["resnet18", "resnet50", "densenet121", "vgg19"]
    return opt


def get_timm_options():
    return timm.list_models(pretrained=False)