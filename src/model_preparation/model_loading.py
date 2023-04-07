import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import transformers

import logging

logger = logging.getLogger(__name__)


@hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
def load_model(cfg: DictConfig):
    try:
        model = instantiate(cfg.model.load)
        logger.info(f'Loaded {cfg.model.name}')
        return model
    except Exception as e:
        logger.exception(f'Couldn\'t load {cfg.model.name}')
        raise e


if __name__ == '__main__':
    try:
        load_model()
    except Exception as e:
        logger.exception(e)
        raise e
