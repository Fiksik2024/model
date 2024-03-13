
import torch

# Зависимости, необходимые для вашей модели
dependencies = ['torch', 'yaml']

def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """
    Universal model creation function for YOLOv5 models.
    """
    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from utils.general import check_img_size, set_logging
    from utils.torch_utils import select_device

    # Set logging
    set_logging(verbose=verbose)

    # Model path
    model_path = 'best.pt' if pretrained and name == 'custom' else name

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(model_path, device=device, fuse=autoshape)

    # Autoshape wrapper
    if autoshape:
        model = AutoShape(model)

    return model

def custom(path='best.pt', channels=3, classes=63, autoshape=True, verbose=True, device=None):
    """
    Custom model loader. Loads a custom YOLOv5 model from a specified path.
    """
    return _create(path, pretrained=True, autoshape=autoshape, verbose=verbose, device=device)
