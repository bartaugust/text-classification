import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import torchtext


class CustomDataLoader:
    def __init__(self, cfg: DictConfig, dataset):
        self.cfg = cfg

        self.dataset = instantiate(cfg.dataset.load[dataset]).shuffle()
        # self.tokenizer = instantiate(cfg.tokenizer.load)

        # self.tokenized_dataset = self.tokenize()
        self.dataloader = self.create_dataloader()

    def tokenize(self):
        tokens = []

        # for label, line in self.dataset:
        #     tokenized_line = self.tokenizer(line)
        #     tokenized_line['labels'] = label
        #     tokens.append(tokenized_line)
        return tokens

    def create_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.cfg.data_loading.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           pin_memory=True,
                                           )

    def split_val(self):
        return self
