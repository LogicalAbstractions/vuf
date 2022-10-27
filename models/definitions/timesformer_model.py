import torch
from timesformer_pytorch import TimeSformer


class TimesFormerModel(TimeSformer):
    def forward(self, x, **kwargs):
        x = torch.stack(x, dim=1)
        return super().forward(x, **kwargs)
