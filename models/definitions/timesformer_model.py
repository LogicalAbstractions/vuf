from typing import Tuple

import torch
from timesformer_pytorch import TimeSformer

from models.encoders.encoder_description import EncoderDescription
from models.model_parameters import ModelParameters


class TimesFormerModel(TimeSformer):
    def forward(self, x, **kwargs):
        x = torch.stack(x, dim=1)
        return super().forward(x, **kwargs)

    @staticmethod
    def splitter(model):
        return None

    @staticmethod
    def hyper_parameters() -> dict:
        return {
            "dim": 128,
            "patch_size": 16,
            "depth": 12,
            "heads": 8,
            "dim_head": 64,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1
        }

    @staticmethod
    def create(model_parameters: ModelParameters):
        return TimesFormerModel(
            dim=model_parameters.hyper_parameters["dim"],
            image_size=model_parameters.image_size[0],
            patch_size=model_parameters.hyper_parameters["patch_size"],
            num_frames=model_parameters.frame_count,
            num_classes=model_parameters.class_count,
            depth=model_parameters.hyper_parameters["depth"],
            heads=model_parameters.hyper_parameters["heads"],
            dim_head=model_parameters.hyper_parameters["dim_head"],
            attn_dropout=model_parameters.hyper_parameters["attn_dropout"],
            ff_dropout=model_parameters.hyper_parameters["ff_dropout"]
        )
