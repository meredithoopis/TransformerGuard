import torch
import logging
from core import train
import os
import hydra
import utils
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.d4rl/datasets"))
    log_dict = utils.get_log_dict()
    for seed in cfg.seeds:
        train(cfg, seed, log_dict, -1, logger, dataset_dir)
    utils.visualize_perf_drop_curve(cfg, log_dict)

if __name__ == "__main__":
    main()

