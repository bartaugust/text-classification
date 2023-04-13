from hydra.utils import instantiate

import torch
from torch import nn

from tqdm import trange, tqdm

import logging
import lightning as L

logger = logging.getLogger(__name__)


class TextClassificationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.base_model = instantiate(cfg.model.load)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.base_model.config.hidden_size, cfg.dataset.classes)

    def forward(self, input_ids, attention_mask):
        bertOutput = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(bertOutput['pooler_output'])

        return self.out(output)


class TextClassification:
    def __init__(self, cfg):
        self.cfg = cfg

        # self.cuda_available = torch.cuda.is_available()
        # self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        self.fabric = L.Fabric(accelerator="cuda")
        self.fabric.launch()

        self.model = TextClassificationModel(cfg)

        # instantiate(cfg.model.load)

        self.criterion = instantiate(self.cfg.model.params.loss)
        self.optimizer = instantiate(self.cfg.model.params.optimizer, self.model.parameters())

        self.model, self.optimizer, self.criterion = self.fabric.setup(self.model, self.optimizer, self.criterion)

        self.compiled_model = torch.compile(self.model)

        # self.tokenizer = instantiate(cfg.tokenizer.load)

    def train(self, train_loader):
        train_loader = self.fabric.setup_dataloaders(train_loader)
        logger.info(f'started training on {self.fabric.device}')
        for epoch in range(self.cfg.model.params.epochs):
            logger.info(f'epoch: {epoch+1}/{self.cfg.model.params.epochs}')
            with tqdm(train_loader, unit="batch", total=len(list(train_loader))) as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}")
                    tokenizer = instantiate(self.cfg.tokenizer.load)

                    tokenized = tokenizer(batch[1], **self.cfg.tokenizer.params)

                    input_ids = tokenized['input_ids'].to(self.fabric.device)
                    attention_mask = tokenized['attention_mask'].to(self.fabric.device)
                    labels = batch[0]-1

                    self.optimizer.zero_grad()

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    loss = self.criterion(outputs, labels)

                    correct = (predictions == labels).sum().item()
                    accuracy = correct / self.cfg.data_loading.batch_size

                    # loss.backward()
                    self.fabric.backward(loss)
                    self.optimizer.step()
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
