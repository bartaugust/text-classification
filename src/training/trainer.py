from typing import Any, cast, Iterable, List, Literal, Optional, Tuple, Union
from collections.abc import Mapping

from hydra.utils import instantiate

import torch
from torchmetrics import ConfusionMatrix

from tqdm import trange, tqdm

import logging
import lightning as L
from lightning.fabric.loggers import TensorBoardLogger

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, cfg, version=None):
        torch.cuda.empty_cache()
        self.cfg = cfg

        # self.cuda_available = torch.cuda.is_available()
        # self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        logger = TensorBoardLogger(root_dir="/home/trabor/PycharmProjects/PJN/logs/tensorboard",
                                   name=f'{cfg.model.name}/{cfg.dataset.name}', version=version)
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

        logging.info(f'started training on {self.fabric.device}')
        self.save_model(
            f'{self.cfg.paths.saved_models}/{self.cfg.version}/model-{self.cfg.model.name}--dataset-{self.cfg.dataset.name}--epoch-0.ckpt')

        for epoch in range(self.cfg.model.params.epochs):
            logging.info(f'epoch: {epoch + 1}/{self.cfg.model.params.epochs}')
            self.train_loop(train_loader, epoch)
            self.val_loop(val_loader, epoch)
            self.save_model(
                f'{self.cfg.paths.saved_models}/{self.cfg.version}/model-{self.cfg.model.name}--dataset-{self.cfg.dataset.name}--epoch-{epoch + 1}.ckpt')
        logging.info('fit finished')

    def train_loop(self, train_loader, epoch):

        self.fabric.call("on_train_epoch_start")
        loader_len = 0
        # loader_len = len(list(train_loader))
        iterable = self.progbar_wrapper(
            train_loader, total=loader_len, desc=f"Epoch {epoch}"
        )
        for batch_idx, batch in enumerate(iterable):
            self.fabric.call("on_train_batch_start", batch, batch_idx)
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
            logs = {"Train loss": loss, "Train accuracy": accuracy}
            self.fabric.log_dict(logs, step=epoch * loader_len + batch_idx)
            # loss.backward()
            self.fabric.backward(loss)
            self.optimizer.step()

            iterable.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        self.fabric.call("on_train_epoch_end")

    def val_loop(self, val_loader, epoch):

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`
        torch.set_grad_enabled(False)
        self.fabric.call("on_validation_epoch_start")

        # loader_len = len(list(val_loader))
        loader_len = 0
        iterable = self.progbar_wrapper(
            val_loader, total=loader_len, desc=f"Epoch {epoch}"
        )
        for batch_idx, batch in enumerate(iterable):
            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            input_ids = batch['input_ids'].to(self.fabric.device)
            attention_mask = batch['attention_mask'].to(self.fabric.device)
            labels = batch['label']

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            loss = self.criterion(outputs, labels)

            correct = (predictions == labels).sum().item()
            accuracy = correct / self.cfg.data_loading.batch_size
            logs = {"Validation Loss": loss, "Validation accuracy": accuracy}
            self.fabric.log_dict(logs, step=epoch * loader_len + batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM

            self.fabric.call("on_validation_batch_end", outputs, batch, batch_idx)

            iterable.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def test(self, test_loader):
        logging.info('test started')
        torch.cuda.empty_cache()
        all_acc = []
        all_loss = []
        all_pred = []
        all_labels = []
        with torch.no_grad():
            self.model.eval()
            # loader_len = len(list(test_loader))
            loader_len = 0
            loader = self.progbar_wrapper(
                test_loader, total=loader_len, desc=f"Epoch {self.cfg.epoch}"
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
                all_loss.append(loss.item())
                all_pred.append(predictions)
                all_labels.append(labels)

                loader.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

        mean_acc = np.mean(all_acc)
        mean_loss = np.mean(all_loss)

        confmat = ConfusionMatrix(task="multiclass", num_classes=3).to(self.fabric.device)
        cm = confmat(torch.cat(all_pred), torch.cat(all_labels))
        cm = cm / cm.sum(axis=1)
        fig = plt.figure()
        sns.heatmap(cm.cpu(), annot=True)
        self.fabric.logger.experiment.add_figure('Confusion matrix', fig, self.cfg.epoch)

        logs = {"Test loss": mean_loss, "Test accuracy": mean_acc}
        self.fabric.log_dict(logs, step=self.cfg.epoch)
        torch.cuda.empty_cache()

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, **kwargs)
        return iterable

    def save_model(self, path):
        state = {'model': self.model, 'optimizer': self.optimizer}
        self.fabric.save(path, state)
        logging.info('saved model')

    def load_model(self, path):
        state = self.fabric.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
