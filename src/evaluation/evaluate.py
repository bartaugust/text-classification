import hydra
from omegaconf import DictConfig

from src.data_preparation.data_loading import load_data
from src.model_preparation.model_loading import load_model
import logging

import torch

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')


@hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
def evaluate(cfg: DictConfig):
    data = load_data(cfg)
    model = load_model(cfg)
    if cfg.model.trained_path:
        model.load_saved()
    else:
        model.fit(data[0].dataloader)
    logger.info('aa')


if __name__ == '__main__':
    try:
        evaluate()
    except Exception as e:
        logger.exception(e)

        raise e
