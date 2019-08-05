from onet import OccupancyNetwork
import torch
from dataloader import get_dataset
from dataloader.core import collate_remove_none, worker_init_fn
from checkpoints import CheckpointIO
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import argparse
from scipy.spatial import Delaunay


def gen_points(cloud, samples):
    '''
    Sample points from uniform distribution in unit qube and calculate the corresponding occupancies to the
    given pointcloud,
    :param cloud: Pointcloud of the specific shape
    :param samples: amount of points to generate
    :return: Tupel of: [0] points, [1] occ
    '''
    points = np.random.uniform([-1, -1, -1], [1, 1, 1], (samples, 3))

    if not isinstance(cloud, Delaunay):
        hull = Delaunay(cloud)
        print(hull.points.shape)
    else:
        hull = cloud
    occ = hull.find_simplex(points)
    occ[occ >= 0] = 1
    occ[occ < 0] = 0
    # shape_occ = np.ones((clound_arr.shape[0]))
    return points, occ


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize latent space of trained model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", nargs=1, metavar="<pen|sphere|qube>", required=True, type=str)
    parser.add_argument("-z", "--z_dim", nargs=1, default=[2], type=int, help="Set the dimension of the latent space")
    parser.add_argument("-v", "--visualize", action='store_true', help="if plot should be visualized")
    args = parser.parse_args()
    current_dir = (os.getcwd())
    voxel_model = args.model[0]
    z_dim = args.z_dim[0]

    if voxel_model not in ["qube", "sphere", "pen"]:
        print("Model not known!")
        exit(0)

    out_path = "out/"
    gen = "generation/"
    data_path = "data/dataset/"
    model_name = 'model' + '_z_dim_' + str(z_dim) + '.pt'
    DATASET_PATH = os.path.join(current_dir, data_path, voxel_model, '')
    print(DATASET_PATH)
    OUT_DIR = os.path.join(current_dir, out_path, voxel_model, '')
    GEN_DIR = os.path.join(OUT_DIR, gen, '')
    print(OUT_DIR)
    if not os.path.exists(GEN_DIR):
        os.makedirs(GEN_DIR)

    # Create torch device for GPU computing
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    # Create/Load model
    occ_net = OccupancyNetwork(device=device, z_dim=z_dim)
    checkpoint_io = CheckpointIO(OUT_DIR, model=occ_net)
    iteration = 0
    try:
        load_dict = checkpoint_io.load(model_name)
        iteration = load_dict
    except FileExistsError:
        print("No model found!")
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)

    test_dataset = get_dataset("test", dataset_path=DATASET_PATH)

    # Create the dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=4, shuffle=False,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)


