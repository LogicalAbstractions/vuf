from typing import Tuple

import torch
from fastai.layers import TimeDistributed
from fastai.torch_core import params
from fastai.vision.learner import create_body, create_head
import torch.nn as nn

from models.encoders.encoder_description import EncoderDescription
from models.model_parameters import ModelParameters


class CnnModel(nn.Module):
    def __init__(self,
                 model_parameters: ModelParameters):
        super().__init__()

        arch = model_parameters.encoder_description()

        self.encoder = TimeDistributed(create_body(arch, pretrained=True), low_mem=True)
        self.dropout = nn.Dropout()
        self.head = TimeDistributed(create_head(model_parameters.encoder_description.feature_count,
                                                model_parameters.class_count))

    def forward(self, x):
        x = torch.stack(x, dim=1)
        x = self.encoder(x)
        x = self.dropout(x)

        return self.head(x).mean(dim=1)

    @staticmethod
    def splitter(model):
        return [params(model.encoder), params(model.head)]

    @staticmethod
    def hyper_parameters() -> dict:
        return dict()
