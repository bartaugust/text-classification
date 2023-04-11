from hydra.utils import instantiate

import torch
from torch import nn

import logging

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
        self.model = TextClassificationModel(cfg)
            # instantiate(cfg.model.load)

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')

        self.criterion = instantiate(self.cfg.model.params.loss)
        self.optimizer = instantiate(self.cfg.model.params.optimizer, self.model.parameters())

        # self.tokenizer = instantiate(cfg.tokenizer.load)

    def train(self, train_loader):
        for epoch in range(self.cfg.model.params.epochs):
            logger.info('started training')
            for batch in train_loader:
                tokenizer = instantiate(self.cfg.tokenizer.load)

                tokenized = tokenizer(batch[1], padding=True, truncation=True, max_length=512, return_tensors='pt')
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                labels = batch[0].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
