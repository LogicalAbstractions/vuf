from typing import Callable
import torch
import torch.nn as nn


class EncoderDescription:
    def __init__(self, id: str, factory: Callable[[], nn.Module], feature_count: int):
        self.id = id
        self.factory = factory
        self.feature_count = feature_count

    def __call__(self, *args, **kwargs) -> nn.Module:
        return self.factory()
