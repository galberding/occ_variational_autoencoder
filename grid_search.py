from onet import OccupancyNetwork
from onet.training import Trainer
import torch
from dataloader import get_dataset
from dataloader.core import collate_remove_none, worker_init_fn
from checkpoints import CheckpointIO
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import argparse
import numpy as np
from torch.nn import init
import datetime
import sys
from train import create_work_dirs, generate_datasets, load_trainer_from_model, train_loop


def parse_args():
    parser = argparse.ArgumentParser(description="Train the network for the given data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", nargs=1, metavar="<pen|sphere|qube>", required=True, type=str)
    parser.add_argument("-z", "--z_dim", nargs=1, default=[128], type=int, help="Set the dimension of the latent space")
    parser.add_argument("-i", "--max_iterations", nargs=1, default=[10000], type=int, help="Set max epoch iteration")
    parser.add_argument("-o", "--output_dir", nargs=1, default=[''], type=str, help="Set output dir")
    parser.add_argument("-b", "--batch", nargs=1, default=[3], type=int, help="Batchsize")
    parser.add_argument("-e", "--error", nargs=1, default=[1], type=int, help="Quit training when error is reached!")

    args = parser.parse_args()

    current_dir = os.getcwd()
    voxel_model = args.model[0]
    z_dim = args.z_dim
    max_iterations = args.max_iterations[0]
    print(max_iterations)
    batch_size = args.batch[0]
    error = args.error[0]
    return args, batch_size, current_dir, max_iterations, voxel_model, z_dim, error




def  main():


    args, batch_size, current_dir, max_iterations, voxel_model, z_dim, error = parse_args()


    for z in z_dim:
        DATASET_PATH, OUT_DIR, model_name = create_work_dirs(args, batch_size, current_dir, voxel_model, z)

        # Create torch device for GPU computing
        is_cuda = (torch.cuda.is_available())
        device = torch.device("cuda" if is_cuda else "cpu")

        # Dataset loading
        test_loader, train_loader, vis_train_loader = generate_datasets(DATASET_PATH, batch_size)

        # Tensorboard initializing


        logger = SummaryWriter(
            os.path.join(OUT_DIR, datetime.datetime.now().strftime('logs_%Y_%m_%d_%H_%M_%S' + model_name[5:-3])))

        # Model and trainer loading
        checkpoint_io, epoch_it, it, trainer = load_trainer_from_model(OUT_DIR, device, model_name, z)


        train_loop(model_name, checkpoint_io, test_loader, train_loader, trainer, vis_train_loader, logger, error=error, max_iterations=max_iterations, epoch_it=epoch_it, it=it)



if __name__ == '__main__':
    main()