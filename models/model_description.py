from typing import Callable, List
import torch.nn as nn

from models.encoders.encoder_description import EncoderDescription


class ModelDescription:
    def __init__(self,
                 id: str,
                 factory: Callable[[dict, EncoderDescription, int], nn.Module],
                 splitter: Callable[[nn.Module], List],
                 hyper_parameters: dict):
        self.id = id
        self.factory = factory
        self.splitter = splitter
        self.hyper_parameters = hyper_parameters

    def __call__(self, hyper_parameters: dict, encoder_description: EncoderDescription, class_count: int) -> nn.Module:
        return self.factory(hyper_parameters, encoder_description, class_count)

    @staticmethod
    def from_class(id: str, cls):
        return ModelDescription(id, lambda hp, e, c: cls(hp, e, c), cls.splitter, cls.hyper_parameters())
