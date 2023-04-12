import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import lightning as L

import transformers

from src.model_preparation.text_classification_model import TextClassificationModel, TextClassification
from src.data_preparation.data_loading import load_data

import logging

logger = logging.getLogger(__name__)




@hydra.main(version_base='1.3', config_path='../../conf', config_name='config')
def load_model(cfg: DictConfig):
    try:
        model = TextClassification(cfg)
        logger.info(f'Loaded model: {cfg.model.name}')
        return model
    except Exception as e:
        logger.exception(f'Couldn\'t load model: {cfg.model.name}')
        raise e


if __name__ == '__main__':
    try:
        load_model()
    except Exception as e:
        logger.exception(e)
        raise e
