import hydra
from omegaconf import DictConfig

from src.training.train_model import train
import logging

import torch

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)

        raise e
# import transformers

# tmp = transformers.DistilBertModel.