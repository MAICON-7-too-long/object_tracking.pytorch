import torch
import numpy as np

import wandb

import os.path as osp
import random
import click

import mmengine
from mmengine.utils import mkdir_or_exist
from mmengine.runner import set_random_seed, Runner

@click.command()
@click.argument("config_file")
def main(config_file = './configs/det/faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py'):

    # Read configure file and change some configs
    # cfg = mmengine.Config.fromfile('./configs/det/faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py')
    cfg = mmengine.Config.fromfile(config_file)
    cfg.data_root = 'data/MOT17_tiny/'
    cfg.train_dataloader.dataset.data_root = 'data/MOT17_tiny/'
    cfg.test_dataloader = cfg.test_cfg = cfg.test_evaluator = None
    cfg.val_dataloader = cfg.val_cfg = cfg.val_evaluator = None
    cfg.visualizer.name = 'mot_visualizer'

    cfg.work_dir = './tutorial_exps/detector'
    cfg.randomness = dict(seed=777, deterministic=True)
    cfg.gpu_ids = range(1)

    # Remove randomness
    torch.cuda.manual_seed(777)
    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f'Config:\n{cfg.pretty_text}')

    # wandb init
    run_name = f'{config["dataset_name"]}_{config["model_name"]}_{datetime.now().strftime("%m_%d-%H_%M_%S")}'

    wandb.login()
    run = wandb.init(
        project="maicon-object-tracking",
        name=run_name,
        config=cfg
    )

    # Train Model
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    runner = Runner.from_cfg(cfg)
    runner.train()

