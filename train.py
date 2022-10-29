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
from models.model_parameters import ModelParameters
from models.model_registry import ModelRegistry


def top_2_accuracy(inp, targ, axis=-1):
    return top_k_accuracy(inp, targ, k=2, axis=axis)


def execute_training():
    #dataset_path = "/media/bglueck/Data/Datasets/soccernet-datasets/4sec-5fps/small/train"
    dataset_path = "/mnt/vol_b/data/4sec-5fps/small/train"

    configuration = {
        "encoder": "resnet18",
        "model": "timesformer",
        "dataset_path": dataset_path,
        "batch_size": 2,
        "frame_size": (192, 192),
        "val_split": 0.3,
        "frozen_epochs": 100,
        "epochs": 200
    }

    print("Training from: {}".format(dataset_path))

    dataset_path = configuration["dataset_path"]

    encoder_registry = EncoderRegistry()
    model_registry = ModelRegistry()

    dataset = VideoDataset(Path(dataset_path))
    encoder_description = encoder_registry[configuration["encoder"]]
    model_description = model_registry[configuration["model"]]

    model_hyper_parameters = model_description.hyper_parameters
    model_hyper_parameters["rnn_layers"] = 1
    model_hyper_parameters["heads"] = 16
    model_hyper_parameters["dim_head"] = 128
    model_hyper_parameters["dim"] = 256

    configuration["model_hyper_parameters"] = model_hyper_parameters
    configuration["classes"] = dataset.get_class_ids()

    wandb.init(project='vuf', config=configuration)

    model_parameters = ModelParameters(model_hyper_parameters, encoder_description, configuration["frame_size"],
                                       len(dataset.get_class_ids()), dataset.get_frame_count())

    model = model_description(model_parameters).cuda()

    learner = Learner(dataset.create_dataloaders(configuration["batch_size"], configuration["frame_size"],
                                                 configuration["val_split"]),
                      model,
                      metrics=[accuracy, error_rate, top_2_accuracy]).to_fp16()

    if model_description.splitter is not None:
        learner.splitter = model_description.splitter

    callbacks = [TensorBoardCallback(trace_model=False),
                 WandbCallback(),
                 SaveModelCallback()]

    if model_description.splitter is not None:
        learner.fine_tune(configuration["epochs"], freeze_epochs=configuration["frozen_epochs"], cbs=callbacks)
    else:
        learner.fit_one_cycle(configuration["epochs"], cbs=callbacks)


if __name__ == '__main__':
    execute_training()
