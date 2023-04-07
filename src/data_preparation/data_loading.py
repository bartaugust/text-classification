import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

import logging

logger = logging.getLogger(__name__)


# @hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
# def load_all_datasets(cfg: DictConfig):
#     for name, cfg_dataset in cfg.datasets.items():
#         try:
#             dataset = instantiate(cfg_dataset.load)
#             logger.info(f'Loaded {name}')
#         except Exception as e:
#             logger.exception(f'Couldn\'t load {name}')
#             raise e

@hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
def load_data(cfg: DictConfig):
    try:
        train_dataset = instantiate(cfg.dataset.load_train)
        test_dataset = instantiate(cfg.dataset.load_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.data_loading.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.data_loading.batch_size, shuffle=False)
        logger.info(f'Loaded {cfg.dataset.name}')
        return train_loader, test_loader
    except Exception as e:
        logger.exception(f'Couldn\'t load {cfg.dataset.name}')
        raise e


if __name__ == '__main__':
    try:
        load_data()
    except Exception as e:
        logger.exception(e)
        raise e
