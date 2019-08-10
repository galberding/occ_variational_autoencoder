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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train the network for the given data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", nargs=1, metavar="<pen|sphere|qube>", required=True, type=str)
    parser.add_argument("-z", "--z_dim", nargs=1, default=[2], type=int, help="Set the dimension of the latent space")
    parser.add_argument("-i", "--max_iterations", nargs=1, default=[10000], type=int, help="Set max epoch iteration")
    parser.add_argument("-c", "--checkpoint", nargs=1, default=[100], type=int,
                        help="Set after how many iterations the model should be saved.")
    parser.add_argument("-e", "--eval", nargs=1, default=[100], type=int, help="Perform the validation every x rounds.")
    parser.add_argument("-b", "--batch", nargs=1, default=[5], type=int, help="Batchsize")
    parser.add_argument("-p", "--path", nargs=1, default=[''], type=str,
                        help="Specify the absolute project path, if not set the current working directory will be choosed")
    parser.add_argument("-v", "--vis", nargs=1, default=[50], type=int, help="visualize after x iterations")
    parser.add_argument("-o", "--output_dir", nargs=1, default=[''], type=str, help="Set output dir")
    # parser.add_argument('--pearson', nargs=1, default=[100], )
    args = parser.parse_args()

    current_dir = args.path[0]
    if not current_dir:
        current_dir = os.getcwd()
    voxel_model = args.model[0]
    z_dim = args.z_dim[0]
    max_iterations = args.max_iterations[0]
    print(max_iterations)
    batch_size = args.batch[0]
    checkpoint_every = args.checkpoint[0]
    eval_network = args.eval[0]
    vis = args.vis[0]
    # pearson = args.pearson

    print(voxel_model)
    print(z_dim)

    if voxel_model not in ["qube", "sphere", "pen"]:
        print("Model not known!")
        exit(0)

    out_path = "out/"
    data_path = "data/dataset/"
    model_name = 'model' + '_z_dim_' + str(z_dim) + '.pt'
    DATASET_PATH = os.path.join(current_dir, data_path, voxel_model, '')
    print(DATASET_PATH)
    if not args.output_dir[0]:
        OUT_DIR = os.path.join(current_dir, out_path, voxel_model, '')
    else:
        OUT_DIR = os.path.join(current_dir, out_path, args.output_dir[0], '')

    print(OUT_DIR)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Create torch device for GPU computing
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    # Load training data
    train_dataset = get_dataset("train", dataset_path=DATASET_PATH)
    # Load validation data
    test_dataset = get_dataset("test", dataset_path=DATASET_PATH)

    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=25, pin_memory=True, shuffle=False,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)

    test_loader_2 = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__(), pin_memory=True, shuffle=False,
    )

    # create the model
    logger = SummaryWriter(os.path.join(OUT_DIR, 'logs'))
    occ_net = OccupancyNetwork(z_dim=z_dim, device=device, logger=logger)
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
    # logger.add_custom_scalars_multilinechart(['pearson_z_0/x', 'pearson_z_0/y', 'pearson_z_0/z'])
    # epoch_it = 0
    # it = 0
    # vis = 50

    # test = ['pearson_z_0/x', 'pearson_z_0/y', 'pearson_z_0/z', 'pearson_z_1/x', 'pearson_z_1/y', 'pearson_z_1/z',
    #  'pearson_z_2/x', 'pearson_z_2/y', 'pearson_z_2/z', 'pearson_z_3/x', 'pearson_z_3/y', 'pearson_z_3/z']
    # logger.add_custom_scalars_multilinechart(['pearson_z_0/x', 'pearson_z_0/y', 'pearson_z_0/z'], title="dim 0")
    # logger.add_custom_scalars_multilinechart(['pearson_z_1/x', 'pearson_z_1/y', 'pearson_z_1/z'], title="dim 1")
    # logger.add_custom_scalars_multilinechart(['pearson_z_3/x', 'pearson_z_3/y', 'pearson_z_3/z'], title="dim 2")
    while epoch_it < max_iterations:
        epoch_it += 1
        for batch in train_loader:
            it += 1
            # print(batch)
            loss = trainer.train_step(batch)
            logger.add_scalar(voxel_model + '/' + 'train/loss', loss, it)
            print("Epoch: ", epoch_it, " Iteration: ", it, " Loss: ", loss)
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                print('Saving checkpoint')
                checkpoint_io.save(model_name, epoch_it=epoch_it, it=it)
            if (eval_network > 0 and (it % eval_network) == 0):
                print("Evaluate network")
                eval_dict = trainer.evaluate(test_loader)
                logger.add_scalar(voxel_model + '/' + 'val/loss', (eval_dict['loss']), it)
                logger.add_scalar(voxel_model + '/' + 'val/rec_error', (eval_dict['rec_error']), it)
                logger.add_scalar(voxel_model + '/' + 'val/kl-div', (eval_dict['kl']), it)
                logger.add_scalar('val/iou', (eval_dict['iou']), it)

            if (it % vis == 0):
                figs = trainer.visualize(test_loader)
                for i, fig in enumerate(figs):
                    logger.add_figure('val/reconstruction/' + str(i), fig, it)

                # zs = trainer.calculate_pearson(test_loader)
                # multiscal_tags = []
                # for k, v in zs.items():
                #     # print(v.keys())
                #
                #     for k2, v2 in v.items():
                #         tag = 'pearson_z_' + str(k) + '/' + k2
                #         # print(v2)
                #         logger.add_scalar(tag, v2[0], it)
                #         multiscal_tags.append(tag)
            # print(multiscal_tags)
            # logger.add_custom_scalars_multilinechart(multiscal_tags, title=str(k), category="test/"+str(k))
