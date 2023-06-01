from typing import Any, cast, Iterable, List, Literal, Optional, Tuple, Union

from hydra.utils import instantiate

import torch
from torch import nn

from tqdm import trange, tqdm

import logging
import lightning as L

logger = logging.getLogger(__name__)


class TextClassificationModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.base_model = instantiate(cfg.model.load)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.base_model.config.hidden_size, cfg.dataset.classes)

    def forward(self, input_ids, attention_mask):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # output = self.drop(output[self.cfg.model.drop])

        # return self.out(output)
        return output[self.cfg.model.last]

