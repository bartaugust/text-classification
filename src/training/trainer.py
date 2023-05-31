from typing import Any, cast, Iterable, List, Literal, Optional, Tuple, Union
from collections.abc import Mapping

from hydra.utils import instantiate

import torch
from torch import nn

from tqdm import trange, tqdm

import logging
import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from lightning_utilities import apply_to_collection

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, cfg):
        torch.cuda.empty_cache()
        self.cfg = cfg

        # self.cuda_available = torch.cuda.is_available()
        # self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        logger = TensorBoardLogger(root_dir="/home/trabor/PycharmProjects/PJN/logs/tensorboard")
        self.fabric = L.Fabric(**self.cfg.trainer.fabric, loggers=logger)
        self.fabric.launch()
        self.model = model

        # instantiate(cfg.model.load)

        self.criterion = instantiate(self.cfg.model.params.loss)
        self.optimizer = instantiate(self.cfg.model.params.optimizer, self.model.parameters())

        self.model, self.optimizer, self.criterion = self.fabric.setup(self.model, self.optimizer, self.criterion)

        # self.compiled_model = torch.compile(self.model)

        self.current_epoch = 0
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self.fabric.log_dict(self._current_train_return)
        # self.tokenizer = instantiate(cfg.tokenizer.load)

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader = None,
            ):

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
        iterable = self.progbar_wrapper(
            train_loader, total=loader_len, desc=f"Epoch {self.current_epoch}"
        )
        for batch_idx, batch in enumerate(iterable):
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
            logs = {"loss": loss, "accuracy": accuracy}
            self.fabric.log_dict(logs,step=batch_idx)
            # loss.backward()
            self.fabric.backward(loss)
            self.optimizer.step()

            self._format_iterable(iterable, self._current_train_return, "train")
            # loader.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        self.fabric.call("on_train_epoch_end")

    def test(self, test_loader):
        torch.cuda.empty_cache()
        with torch.no_grad():
            all_acc = []
            all_loss = []
            self.model.eval()
            loader_len = len(list(test_loader))
            loader = self.progbar_wrapper(
                test_loader, total=loader_len, desc=f"Epoch {self.current_epoch}"
            )
            for batch_idx, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.fabric.device)
                attention_mask = batch['attention_mask'].to(self.fabric.device)
                labels = batch['label'].to(self.fabric.device)

                self.fabric.call("on_before_zero_grad", self.optimizer)
                self.optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                loss = self.criterion(outputs, labels)

                correct = (predictions == labels).sum().item()
                accuracy = correct / self.cfg.data_loading.batch_size

                all_acc.append(accuracy)
                all_loss.append(loss)
                loader.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            return all_acc, all_loss

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def save_model(self, path):
        state = {'model': self.model, 'optimizer': self.optimizer}
        self.fabric.save(path, state)

    def load_model(self, path):
        state = self.fabric.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.
        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.
        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
