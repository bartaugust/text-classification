import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import torchtext


class CustomDataLoader:
    def __init__(self, cfg: DictConfig, dataset):
        self.cfg = cfg

        self.dataset = dataset
        self.tokenizer = instantiate(cfg.tokenizer.load)

        # self.tokenized_dataset = self.tokenize()
        self.dataloader = self.create_dataloader()

    def create_dataloader(self):
        def tokenize(batch):
            all_lists = {
                'input_ids': [],
                'attention_mask': [],
                # 'token_type_ids':[],
                'label': []
            }

            for elem in batch:
                all_lists['label'].append(elem[0] - 1)

                tokenized = self.tokenizer(elem[1], **self.cfg.tokenizer.params)
                all_lists['attention_mask'].append(tokenized['attention_mask'])
                all_lists['input_ids'].append(tokenized['input_ids'])
                # all_lists['token_type_ids'].append(tokenized['token_type_ids'])

            all_lists['attention_mask'] = torch.stack(all_lists['attention_mask']).squeeze()
            all_lists['input_ids'] = torch.stack(all_lists['input_ids']).squeeze()
            # all_lists['token_type_ids'] = torch.stack(all_lists['token_type_ids']).squeeze()
            all_lists['label'] = torch.Tensor(all_lists['label']).type(torch.LongTensor)
            return all_lists

        return torch.utils.data.DataLoader(self.dataset,
                                           collate_fn=tokenize,
                                           batch_size=self.cfg.data_loading.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           pin_memory=True,
                                           )


