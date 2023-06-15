import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

import logging

from src.data_preparation.custom_data_loader import CustomDataLoader

logger = logging.getLogger(__name__)


def split(dataset, split):

    train_split, test_split = torch.utils.data.random_split(dataset, [split, 1 - split])
    return dataset, dataset

@hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
def load_data(cfg: DictConfig):
    try:
        # train_dataset, test_dataset = load_all_datasets(cfg)

        train_dataset = instantiate(cfg.dataset.load.train).shuffle()
        test_dataset = instantiate(cfg.dataset.load.test).shuffle()
        if 'val' in cfg.dataset.load.keys():
            val_dataset = instantiate(cfg.dataset.load.val).shuffle()
        else:
            test_dataset, val_dataset = split(test_dataset, cfg.dataset.val_split)

        test_loader = CustomDataLoader(cfg, test_dataset)
        val_loader = CustomDataLoader(cfg, val_dataset)
        train_loader = CustomDataLoader(cfg, train_dataset)

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
