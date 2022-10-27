from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet152, ResNet152_Weights

from models.encoders.encoder_description import EncoderDescription


class EncoderRegistry(dict):

    def __init__(self):
        super().__init__()
        encoders = [
            EncoderDescription("resnet18", lambda: resnet18(ResNet18_Weights.IMAGENET1K_V1), 512),
            EncoderDescription("resnet34", lambda: resnet34(ResNet34_Weights.IMAGENET1K_V1), 512),
            EncoderDescription("resnet152", lambda: resnet152(ResNet152_Weights.IMAGENET1K_V1), 2048)
        ]

        for encoder in encoders:
            self[encoder.id] = encoder
