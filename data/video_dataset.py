from pathlib import Path
from typing import List

from fastai.data.core import DataLoaders, Datasets
from fastai.data.transforms import RandomSplitter, parent_label, Categorize, ToTensor, IntToFloatTensor, Normalize
from fastai.vision.augment import Resize
from fastai.vision.core import imagenet_stats
from fastcore.basics import patch
from fastcore.transform import DisplayedTransform
import torch

from data.video_transforms import VideoDataTransform, VideoStackTransform


@patch
def ls_sorted(self: Path):
    return self.ls().sorted(key=lambda f: int(f.with_suffix('').name))


class VideoDataset:

    def __init__(self, root_path: Path):
        self.root_path = root_path

        self.instances_paths = list()
        class_paths = root_path.ls()

        for class_path in class_paths:
            frame_paths = [x for x in class_path.ls() if x.is_dir()]
            self.instances_paths = self.instances_paths + frame_paths

    def get_class_ids(self) -> List[str]:
        class_paths = self.root_path.ls()
        return [c.name.lower() for c in class_paths]

    def get_frame_count(self) -> int:
        return len(self.instances_paths[0].ls())

    def create_dataloaders(self, batch_size: int, frame_size: int, val_split: float, **kwargs) -> DataLoaders:
        splits = RandomSplitter(val_split)(self.instances_paths)
        data_transform = VideoDataTransform(self.get_frame_count())
        datasets = Datasets(self.instances_paths,
                            tfms=[[data_transform], [parent_label, Categorize]],
                            splits=splits)

        return datasets.dataloaders(batch_size,
                                    after_item=[Resize(frame_size), ToTensor],
                                    after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)],
                                    drop_last=True,
                                    **kwargs)
