import torch
from fastai.layers import TimeDistributed, Flatten, LinBnDrop
from fastai.torch_core import params

import torch.nn as nn
from fastai.vision.learner import create_body

from models.encoders.encoder_description import EncoderDescription


class RnnModel(nn.Module):
    def __init__(self,
                 hyper_parameters: dict,
                 encoder_description: EncoderDescription,
                 class_count: int):
        super().__init__()
        arch = encoder_description()

        self.encoder = TimeDistributed(
            nn.Sequential(create_body(arch, pretrained=True), nn.AdaptiveAvgPool2d(1), Flatten()))
        self.dropout1 = nn.Dropout()

        rnn_layers = hyper_parameters["rnn_layers"]
        rnn_feature_count = hyper_parameters["rnn_feature_count"]

        self.rnn = nn.LSTM(encoder_description.feature_count, rnn_feature_count, num_layers=rnn_layers,
                           batch_first=True)
        self.dropout2 = nn.Dropout()
        self.head = LinBnDrop(rnn_layers * rnn_feature_count, class_count)

    def forward(self, x):
        x = torch.stack(x, dim=1)
        x = self.encoder(x)
        x = self.dropout1(x)

        bs = x.shape[0]
        _, (h, _) = self.rnn(x)
        h = self.dropout2(h)
        return self.head(h.view(bs, -1))

    @staticmethod
    def splitter(model):
        return [params(model.encoder), params(model.rnn) + params(model.head)]

    @staticmethod
    def hyper_parameters() -> dict:
        return {"rnn_layers": 1, "rnn_feature_count": 512}
