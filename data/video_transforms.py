from pathlib import Path

import torch
from fastai.torch_core import TensorImage
from fastai.vision.core import PILImage
from fastcore.transform import Transform, DisplayedTransform

from data.video_data import VideoData


class VideoDataTransform(Transform):

    def __init__(self, max_frames: int):
        self.max_frames = max_frames

    def encodes(self, path: Path):
        frames = path.ls_sorted()
        num_frames = len(frames)
        sequence_frames = slice(0, min(self.max_frames, num_frames))
        return VideoData(tuple(PILImage.create(f) for f in frames[sequence_frames]))


class VideoStackTransform(DisplayedTransform):

    def encodes(self, tensor):
        return tensor
