from dataclasses import dataclass
from typing import Tuple

from models.encoders.encoder_description import EncoderDescription


@dataclass
class ModelParameters:
    hyper_parameters: dict
    encoder_description: EncoderDescription
    image_size: Tuple[int, int]
    class_count: int
    frame_count: int
