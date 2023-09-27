import sys

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket
from typing import Union, Optional
from training.hybrid_supervised_full_trainer import  hybrid_full_supervised_trainer
from training.full_supervised_tumour_trainer import full_supervised_tumour_trainer
from training.base_trainer import base_trainer
# from training.abdomen_location_trainer import abdomen_location_trainer
from training.hybrid_supervised_abdomen_trainer import  hybrid_supervised_abdomen_trainer
from training.ema_hybrid_supervised_abdomen_trainer import ema_hybrid_supervised_abdomen_trainer
from training.weakly_supervised_tumour_trainer import weakly_supervised_tumour_trainer
from training.semi_supervised_tumour_trainer import semi_supervised_tumour_trainer
from training.abdomen_local_trainer import abdomen_local_trainer
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from torch.backends import cudnn
from datetime import datetime
import wandb
import yaml
from training.load_pretrained_weights import load_pretrained_weights

def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, world_size,pretrained_weights_file):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    # trainer = trainer(experiment_id,fold,num_process,config)
    # if pretrained_weights_file is not None:
    #         if not trainer.was_initialized:
    #             trainer.initialize()
    #         load_pretrained_weights(trainer.network, pretrained_weights_file, verbose=True)
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    # trainer.run_training()

    cleanup_ddp()


def run_training(num_gpus, experiment_tag,fold, wandb_mode,num_process, config, trainer_type):
    if trainer_type == "base":
        trainer = base_trainer
    elif trainer_type == "HSAS":
        trainer = hybrid_supervised_abdomen_trainer
    elif trainer_type == "ESAS":
        trainer = ema_hybrid_supervised_abdomen_trainer
    elif trainer_type == "SSTS":
        trainer = semi_supervised_tumour_trainer
    elif trainer_type == "WSTS":
        trainer = weakly_supervised_tumour_trainer
    elif trainer_type == "ALT":
        trainer = abdomen_local_trainer



    experiment_id = f"{fold}_{trainer_type}_{experiment_tag}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    wandb.init(project=f"FLARE2023_{trainer_type}", name=experiment_id, mode=wandb_mode,config =config)
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if num_gpus > 1:

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     num_gpus,experiment_id,fold,num_process,config),
                 nprocs=num_gpus,
                 join=True)
    else:
        trainer = trainer(experiment_id,fold,num_process,config)
        # if config['Trainer']['pretrained_weights_file'] is not None:
        #     if not trainer.was_initialized:
        #         trainer.initialize()
        #     load_pretrained_weights(trainer.network, config['Trainer']['pretrained_weights_file'], verbose=True)
        #     if trainer.teacher_network is not None:
        #         load_pretrained_weights(trainer.teacher_network, config['Trainer']['pretrained_weights_file'], verbose=True)
        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        trainer.run_training()


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument('-et', '--experiment_tag', type=str, required=True,
                        help='Experiment tag.')
    parser.add_argument('-wm', '--wandb_mode', type=str, required=False, default="offline",
                        help='Wandb mode, online or offline.')
    parser.add_argument('-fold', type=str,required=True,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4 or all')
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/one_stage_config.yaml",
                        help='Config file.')
    parser.add_argument('-np', type=int, default=8, required=False,
                        help='[OPTIONAL] Number of processes for data argmentation. Default: 8')
    parser.add_argument('-tt', type=str, required=False, default="FSAL",
                        help='trainer type: FSAL (First stage abdomen_location_trainer), HSAS (Hybrid_supervised_abdomen_trainer), \
                              SSTS (semi_supervised_tumour_trainer) or HSFS (Hybrid_supervised_full_trainer).')
 
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)
    assert args.tt in ['base','HSAS','ESASs','SSTS','WSTS','ALT']
    run_training(args.num_gpus, args.experiment_tag, args.fold, args.wandb_mode, args.np, CFG,args.tt)


if __name__ == '__main__':
    run_training_entry()
