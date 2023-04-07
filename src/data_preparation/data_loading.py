import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import logging
import torch

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
        dataset = instantiate(cfg.dataset.load)
        logger.info(f'Loaded {cfg.dataset.name}')
        return dataset
    except Exception as e:
        logger.exception(f'Couldn\'t load {cfg.dataset.name}')
        raise e


if __name__ == '__main__':
    try:
        load_data()
    except Exception as e:
        logger.exception(e)
        raise e
