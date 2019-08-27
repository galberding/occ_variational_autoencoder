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


def glorot_weight_zero_bias(model):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.
    Taken from: https://github.com/robintibor/braindecode/blob/master/braindecode/torch_ext/init.py
    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                init.xavier_uniform_(module.weight, gain=1)
            else:
                init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                init.constant_(module.bias, 0)


def main():
    # Argument parser
    args, batch_size, checkpoint_every, current_dir, eval_network, max_iterations, pears, vis, voxel_model, z_dim, gamma = parse_args()

    DATASET_PATH, OUT_DIR, model_name = create_work_dirs(args, batch_size, current_dir, voxel_model, z_dim[0])

    # Create torch device for GPU computing
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    # Dataset loading
    test_loader, train_loader, vis_train_loader = generate_datasets(DATASET_PATH, batch_size)

    # Tensorboard initializing
    logger = SummaryWriter(
        os.path.join(OUT_DIR, datetime.datetime.now().strftime('logs_%Y_%m_%d_%H_%M_%S' + model_name[5:-3])))

    # Model and trainer loading
    checkpoint_io, epoch_it, it, trainer = load_trainer_from_model(OUT_DIR, device, model_name, z_dim[0])

    # train_loop(model_name, checkpoint_io, test_loader, train_loader, trainer, vis_train_loader, logger, vis=1)

    train_loop(model_name, checkpoint_io, test_loader, train_loader, trainer, vis_train_loader, logger, error=5,
               max_iterations=max_iterations, epoch_it=epoch_it, it=it, checkpoint_every=eval_network,
               eval_network=eval_network, pears=eval_network, vis=eval_network)

    # Training
    # train_loop(checkpoint_every, checkpoint_io, epoch_it, eval_network, it, logger, max_iterations, model_name, pears,
    #            test_loader, train_loader, trainer, vis, vis_train_loader)


def create_work_dirs(args, batch_size, current_dir, voxel_model, z_dim):
    '''
    Create directories for storing the model and logs of tensorboard.
    :param args:
    :param batch_size:
    :param current_dir:
    :param voxel_model:
    :param z_dim:
    :return:
    '''
    if voxel_model not in ["qube", "sphere", "pen"]:
        print("Model not known!")
        exit(0)
    out_path = "out/"
    data_path = "data/dataset/"
    model_name = 'model' + '_z_dim_' + str(z_dim) + '_batch_' + str(batch_size) + '.pt'
    DATASET_PATH = os.path.join(current_dir, data_path, voxel_model, '')
    print(DATASET_PATH)
    if not args.output_dir[0]:
        OUT_DIR = os.path.join(current_dir, out_path, voxel_model, '')
    else:
        OUT_DIR = os.path.join(current_dir, out_path, args.output_dir[0], '')
    print(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    return DATASET_PATH, OUT_DIR, model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Train the network for the given data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", nargs=1, metavar="<pen|sphere|qube>", required=True, type=str)
    parser.add_argument("-z", "--z_dim", nargs=1, default=[2], type=int, help="Set the dimension of the latent space")
    parser.add_argument("-i", "--max_iterations", nargs=1, default=[10000], type=int, help="Set max epoch iteration")
    parser.add_argument("-c", "--checkpoint", nargs=1, default=[100], type=int,
                        help="Set after how many iterations the model should be saved.")
    parser.add_argument("-e", "--eval", nargs=1, default=[1000], type=int, help="Perform the validation every x rounds.")
    parser.add_argument("-b", "--batch", nargs=1, default=[25], type=int, help="Batchsize")
    parser.add_argument("-p", "--path", nargs=1, default=[''], type=str,
                        help="Specify the absolute project path, if not set the current working directory will be choosed")
    parser.add_argument("-v", "--vis", nargs=1, default=[50], type=int, help="visualize after x iterations")
    parser.add_argument("-o", "--output_dir", nargs=1, default=[''], type=str, help="Set output dir")
    parser.add_argument("--pearson", nargs=1, default=[100], type=int,
                        help="Set amount of iterations when to calculate pearson")
    parser.add_argument('--gamma', nargs=1, default=[0.6], help="Weight factor for binary cross entropy (0,1). Set low to penalize FP, set hight to penalize FN more. (Not yet implemented)")
    # parser.add_argument('--pearson', nargs=1, default=[100], )
    args = parser.parse_args()
    current_dir = args.path[0]
    if not current_dir:
        current_dir = os.getcwd()
    voxel_model = args.model[0]
    z_dim = args.z_dim
    max_iterations = args.max_iterations[0]
    print(max_iterations)
    batch_size = args.batch[0]
    checkpoint_every = args.checkpoint[0]
    eval_network = args.eval[0]
    vis = args.vis[0]
    pears = args.pearson[0]
    gamma = args.gamma[0]
    return args, batch_size, checkpoint_every, current_dir, eval_network, max_iterations, pears, vis, voxel_model, z_dim, gamma


def train_loop(model_name, checkpoint_io, test_loader, train_loader, trainer, vis_train_loader, logger,
               checkpoint_every=500, epoch_it=0, eval_network=500, it=0, max_iterations=5000, pears=500,
               vis=500, error=0, gamma=0.9):
    '''
    Execute the training for the given parameters
    :param checkpoint_every:
    :param checkpoint_io:
    :param epoch_it:
    :param eval_network:
    :param it:
    :param logger:
    :param max_iterations:
    :param model_name:
    :param pears:
    :param test_loader:
    :param train_loader:
    :param trainer:
    :param vis:
    :param vis_train_loader:

    :return:
    '''

    loss = sys.maxsize
    while epoch_it < max_iterations or error > loss:
        epoch_it += 1
        for batch in train_loader:
            it += 1
            # print(batch)
            loss = trainer.train_step(batch)
            logger.add_scalar('train/loss', loss, it)
            print("Epoch: ", epoch_it, " Iteration: ", it, " Loss: ", loss)
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                print('Saving checkpoint')
                checkpoint_io.save(model_name, epoch_it=epoch_it, it=it)
            if (eval_network > 0 and (it % eval_network) == 0):
                eval_dict = trainer.evaluate(test_loader)
                logger.add_scalar('val/loss', (eval_dict['loss']), it)
                logger.add_scalar('val/rec_error', (eval_dict['rec_error']), it)
                logger.add_scalar('val/kl-div', (eval_dict['kl']), it)
                logger.add_scalar('val/iou', (eval_dict['iou']), it)
                print("Evaluated network")

            if (it % vis == 0):
                figs_test = trainer.visualize(test_loader)
                figs_train = trainer.visualize(vis_train_loader)

                for i, fig in enumerate(figs_test):
                    logger.add_figure('val/reconstruction/' + str(i), fig, it)

                for i, fig in enumerate(figs_train):
                    logger.add_figure('train/reconstruction/' + str(i), fig, it)
                # TODO: consider to call together with pears calculation to only oce cycle through the test set
                latent_vis = trainer.vis_latent_attributes(test_loader)
                for i, tag in enumerate(['']):
                    logger.add_figure('val/latent_attr_vis/' + tag, latent_vis[i], it)

            if (it % pears == 0):
                # zs = trainer.calculate_pearson(test_loader)
                zs = trainer.calculate_pears(test_loader)
                print(zs)
                multiscal_tags = []
                for k, v in zs.items():
                    for k2, v2 in v.items():
                        tag = 'pearson_z_' + str(k) + '/' + k2
                        logger.add_scalar(tag, v2, it)
                        multiscal_tags.append(tag)


def load_trainer_from_model(OUT_DIR, device, model_name, z_dim):
    '''
    Create Network, model saver and trainer.
    :param OUT_DIR: Where the model is stored or where it should be stored if there is none
    :param device: torch device
    :param model_name: the namen under which the model gets saved
    :param z_dim: dimesnianality of latent space
    :return: model loader/saver, current epoch, iteration, trainer for the model
    '''
    occ_net = OccupancyNetwork(z_dim=z_dim, device=device)
    glorot_weight_zero_bias(occ_net)
    optimizer = optim.Adam(occ_net.parameters(), lr=1e-4)
    # Restore the model
    checkpoint_io = CheckpointIO(OUT_DIR, model=occ_net, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load(model_name)
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    # metric_val_best = load_dict.get(
    #     'loss_val_best', -model_selection_sign * np.inf)
    # TODO: Validation scores for the model
    # Write to tensorboard
    trainer = Trainer(occ_net, optimizer, device=device)
    return checkpoint_io, epoch_it, it, trainer


def generate_datasets(DATASET_PATH, batch_size):
    '''
    Load train, val and test dataset.
    :param DATASET_PATH: Path to dataset
    :param batch_size: batch size obviously ...
    :return:
    '''
    # Load training data
    train_dataset = get_dataset("train", dataset_path=DATASET_PATH)
    # Load validation data
    test_dataset = get_dataset("test", dataset_path=DATASET_PATH)
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=2, shuffle=True,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=2, shuffle=False,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)
    vis_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, num_workers=2, shuffle=False,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)
    return test_loader, train_loader, vis_train_loader


if __name__ == '__main__':
    main()
