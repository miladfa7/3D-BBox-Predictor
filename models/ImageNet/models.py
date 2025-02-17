

def build_model(model_name):
    if model_name=="resnet":
        from .resnet import ResNetModel
        model = ResNetModel()
    elif model_name=="vitnet":
        from .vitnet import ViTModel
        model = ViTModel()
    return model
