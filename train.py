from datetime import time, datetime
from pathlib import Path

import wandb
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.learner import Learner
from fastai.metrics import accuracy, error_rate, top_k_accuracy

from data.video_dataset import VideoDataset
from models.encoders.encoder_registry import EncoderRegistry
from models.model_registry import ModelRegistry


def top_2_accuracy(inp, targ, axis=-1):
    return top_k_accuracy(inp, targ, k=2, axis=axis)


def execute_training():
    configuration = {
        "encoder": "resnet18",
        "model": "rnn",
        "dataset_path": "/media/bglueck/Data/Datasets/soccernet-datasets/4sec/large/train",
        "batch_size": 2,
        "frame_size": 256,
        "val_split": 0.3,
        "frozen_epochs": 60,
        "epochs": 60
    }

    dataset_path = configuration["dataset_path"]

    encoder_registry = EncoderRegistry()
    model_registry = ModelRegistry()

    dataset = VideoDataset(Path(dataset_path))
    encoder_description = encoder_registry[configuration["encoder"]]
    model_description = model_registry[configuration["model"]]

    model_hyper_parameters = model_description.hyper_parameters
    model_hyper_parameters["rnn_layers"] = 1

    configuration["model_parameters"] = model_hyper_parameters
    configuration["classes"] = dataset.get_class_ids()

    wandb.init(project='vuf', config=configuration)

    model = model_description(model_hyper_parameters, encoder_description, len(dataset.get_class_ids()))

    learner = Learner(dataset.create_dataloaders(configuration["batch_size"], configuration["frame_size"],
                                                 configuration["val_split"]),
                      model,
                      metrics=[accuracy, error_rate, top_2_accuracy],
                      splitter=model_description.splitter)

    callbacks = [TensorBoardCallback(trace_model=False),
                 WandbCallback(),
                 SaveModelCallback()]

    learner.fine_tune(configuration["epochs"], freeze_epochs=configuration["frozen_epochs"], cbs=callbacks)


if __name__ == '__main__':
    execute_training()
