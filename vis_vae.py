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
from sklearn.decomposition import PCA


def set_subplot_colormap(axes, samples, attr, cmap="hot_r", title="Title", xlable="X", ylable="Y"):
    axes.set_facecolor('#e0e0e0')
    axes.set_title(title)
    axes.set_xlabel(xlable)
    axes.set_ylabel(ylable)
    scatter = axes.scatter(samples[:, 0], samples[:, 1], cmap=cm.get_cmap(cmap), c=attr, s=30)
    plt.colorbar(scatter, ax=axes)


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
    plots = "latent_plots/"
    data_path = "data/dataset/"
    model_name = 'model' + '_z_dim_' + str(z_dim) + '.pt'
    DATASET_PATH = os.path.join(current_dir, data_path, voxel_model, '')
    print(DATASET_PATH)
    OUT_DIR = os.path.join(current_dir, out_path, voxel_model, '')
    PLOT_DIR = os.path.join(OUT_DIR, plots, '')
    print(OUT_DIR)
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

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
    occ_net.eval()

    test_dataset = get_dataset("train", dataset_path=DATASET_PATH)

    # Create the dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=4, shuffle=False,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)

    # Collect attributes of test samples and project them in latent space
    samples = []
    sizes = []
    yaw = []
    pitch = []
    roll = []
    for batch in test_loader:
        # print(batch["inputs"])
        # break
        print(occ_net.infer_z(None, None, batch["inputs"]).sample([1]), " size: ", batch["size"].numpy())
        sizes.append(batch["size"].numpy()[0])
        yaw.append(batch["yaw"].numpy()[0])
        pitch.append(batch["pitch"].numpy()[0])
        roll.append(batch["roll"].numpy()[0])
        samples.append(occ_net.infer_z(None, None, batch["inputs"]).sample([1]).numpy()[0, 0])
    samples = (np.array(samples))
    sizes = np.array(sizes)

    # Check if samples are high dimensional, if so, project them to 2-dims:
    if z_dim > 2:
        samples_pca = PCA(n_components=2).fit_transform(samples)
        samples_tsne = TSNE(n_components=2).fit_transform(samples)
    elif z_dim == 1:
        print("Unsupported dim of latent space!")
        exit(0)

    # samples = TSNE(n_components=2).fit_transform(samples)
    # print(samples.shape)

    fig, axes = plt.subplots(2, 4, figsize=(20, 20))
    set_subplot_colormap(axes[0, 0], samples_pca, sizes, title="Sizes", cmap="summer_r")
    set_subplot_colormap(axes[0, 1], samples_pca, yaw, title="Yaw", cmap="bwr")
    set_subplot_colormap(axes[0, 2], samples_pca, pitch, title="Pitch", cmap="bwr")
    set_subplot_colormap(axes[0, 3], samples_pca, roll, title="Roll", cmap="bwr")

    set_subplot_colormap(axes[1, 0], samples_tsne, sizes, title="Sizes", cmap="summer_r")
    set_subplot_colormap(axes[1, 1], samples_tsne, yaw, title="Yaw", cmap="bwr")
    set_subplot_colormap(axes[1, 2], samples_tsne, pitch, title="Pitch", cmap="bwr")
    set_subplot_colormap(axes[1, 3], samples_tsne, roll, title="Roll", cmap="bwr")
    fig.savefig(os.path.join(PLOT_DIR, "lat_vis_{}_it_{}_dim_{}.pdf".format(voxel_model, it, z_dim)), bbox_inches='tight')

    if args.visualize:
        plt.show()
