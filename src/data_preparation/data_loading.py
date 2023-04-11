import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

import logging

from src.data_preparation.custom_data_loader import CustomDataLoader

logger = logging.getLogger(__name__)


def dataset_split(dataset, val_split):
    # TODO: add  split
    val_dataset = dataset
    return dataset, val_dataset



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
        # train_dataset, test_dataset = load_all_datasets(cfg)
        train_loader = CustomDataLoader(cfg, 'train')
        if 'val' in cfg.dataset.load.keys():
            val_loader = CustomDataLoader(cfg, 'val')
        else:
            val_loader = train_loader.split_val()
        test_loader = CustomDataLoader(cfg, 'test')

        logger.info(f'Loaded dataset: {cfg.dataset.name}')
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.exception(f'Couldn\'t load dataset: {cfg.dataset.name}')
        raise e


if __name__ == '__main__':
    try:
        load_data()
    except Exception as e:
        logger.exception(e)
        raise e
