from models.definitions.cnn_model import CnnModel
from models.definitions.rnn_model import RnnModel
from models.definitions.timesformer_model import TimesFormerModel
from models.model_description import ModelDescription


class ModelRegistry(dict[str, ModelDescription]):
    def __init__(self):
        super().__init__()

        models = [
            ModelDescription.from_class("cnn", CnnModel),
            ModelDescription.from_class("rnn", RnnModel),
            ModelDescription("timesformer", TimesFormerModel.create, None, TimesFormerModel.hyper_parameters())
        ]

        for model in models:
            self[model.id] = model
