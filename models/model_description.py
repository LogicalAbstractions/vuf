from typing import Callable, List, Tuple, Optional
import torch.nn as nn

from models.encoders.encoder_description import EncoderDescription
from models.model_parameters import ModelParameters


class ModelDescription:
    def __init__(self,
                 id: str,
                 factory: Callable[[ModelParameters], nn.Module],
                 splitter: Optional[Callable[[nn.Module], List]],
                 hyper_parameters: dict):
        self.id = id
        self.factory = factory
        self.splitter = splitter
        self.hyper_parameters = hyper_parameters

    def __call__(self,
                 model_parameters: ModelParameters) -> nn.Module:
        return self.factory(model_parameters)

    @staticmethod
    def from_class(id: str, cls):
        return ModelDescription(id, lambda mp: cls(mp), cls.splitter, cls.hyper_parameters())
