from typing import Any, cast, Iterable, List, Literal, Optional, Tuple, Union

from hydra.utils import instantiate

import torch
from torch import nn

from tqdm import trange, tqdm

import logging
import lightning as L

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, cfg):
        torch.cuda.empty_cache()
        self.cfg = cfg

        # self.cuda_available = torch.cuda.is_available()
        # self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        self.fabric = L.Fabric(**self.cfg.trainer.fabric)

        self.model = model

        # instantiate(cfg.model.load)

        self.criterion = instantiate(self.cfg.model.params.loss)
        self.optimizer = instantiate(self.cfg.model.params.optimizer, self.model.parameters())

        self.model, self.optimizer, self.criterion = self.fabric.setup(self.model, self.optimizer, self.criterion)

        # self.compiled_model = torch.compile(self.model)

        self.current_epoch = 0
        # self.tokenizer = instantiate(cfg.tokenizer.load)

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader = None,
            ):
        self.fabric.launch()

        train_loader = self.fabric.setup_dataloaders(train_loader)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader)

        logger.info(f'started training on {self.fabric.device}')

        for epoch in range(self.cfg.model.params.epochs):
            logger.info(f'epoch: {epoch + 1}/{self.cfg.model.params.epochs}')
            self.train_loop(train_loader)

    def train_loop(self, train_loader):
        self.fabric.call("on_train_epoch_start")
        loader_len = len(list(train_loader))
        loader = self.progbar_wrapper(
            train_loader, total=loader_len, desc=f"Epoch {self.current_epoch}"
        )
        for batch_idx, batch in enumerate(loader):
            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # tokenizer = instantiate(self.cfg.tokenizer.load)

            # tokenized = tokenizer(batch[1], **self.cfg.tokenizer.params)

            # input_ids = tokenized['input_ids'].to(self.fabric.device)
            # attention_mask = tokenized['attention_mask'].to(self.fabric.device)
            input_ids = batch['input_ids'].to(self.fabric.device)
            attention_mask = batch['attention_mask'].to(self.fabric.device)
            labels = batch['label']

            self.fabric.call("on_before_zero_grad", self.optimizer)
            self.optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            loss = self.criterion(outputs, labels)

            correct = (predictions == labels).sum().item()
            accuracy = correct / self.cfg.data_loading.batch_size

            # loss.backward()
            self.fabric.backward(loss)
            self.optimizer.step()

            loader.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        self.fabric.call("on_train_epoch_end")

    def predict(self, loader):
        self.model.eval()
        for batch_idx, batch in enumerate(loader):
            pass

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable
