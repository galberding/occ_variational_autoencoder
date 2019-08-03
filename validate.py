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

OUT_DIR = "out/pen"
current_home = ""
torch.initial_seed = 42

# Create torch device for GPU computing
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

occ_net = OccupancyNetwork(device=device)
# optimizer = optim.Adam(occ_net.parameters(), lr=1e-4)
checkpoint_io = CheckpointIO(OUT_DIR, model=occ_net)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    print("No model found!")

test_dataset = get_dataset("test", dataset_path=current_home + "data/dataset/pen/")

# Create the dataloader
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=4, shuffle=False,
    collate_fn=collate_remove_none,
    worker_init_fn=worker_init_fn)

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
    samples.append(occ_net.infer_z(None, None, batch["inputs"]).sample([1]).numpy()[0,0])
samples = (np.array(samples))
sizes = np.array(sizes)

print(sizes)
# Sizes
# plt.scatter(samples[:,0], samples[:,1], cmap=cm.get_cmap("hot_r"), c=sizes)
# Yaw

def set_subplot_colormap(axes, samples, attr, cmap="hot_r", title="Title", xlable="X", ylable="Y"):
    axes.set_facecolor('#e0e0e0')
    axes.set_title(title)
    axes.set_xlabel(xlable)
    axes.set_ylabel(ylable)
    scatter = axes.scatter(samples[:, 0], samples[:, 1], cmap=cm.get_cmap(cmap), c=attr)
    plt.colorbar(scatter, ax=axes)

fig, axes = plt.subplots(2, 2, figsize=(10,10))
# plt.colorbar(axes[0,0].scatter(samples[:,0], samples[:,1], cmap=cm.get_cmap("Greens"), c=sizes), ax=axes[0,0])
set_subplot_colormap(axes[0,0], samples, sizes, title="Sizes", cmap="summer_r")
set_subplot_colormap(axes[0,1], samples, yaw, title="Yaw", cmap="bwr")
set_subplot_colormap(axes[1,0], samples, pitch, title="Pitch", cmap="bwr")
set_subplot_colormap(axes[1,1], samples, roll, title="Roll", cmap="bwr")
# plt.colorbar(axes[0,1].scatter(samples[:,0], samples[:,1], cmap=cm.get_cmap("tab10"), c=yaw), ax=axes[0,1])
# axes[0,1].scatter(samples[:,0], samples[:,1], cmap=cm.get_cmap("tab10"), c=yaw)
# axes[1,0].scatter(samples[:,0], samples[:,1], cmap=cm.get_cmap("tab10"), c=pitch)
# axes[1,1].scatter(samples[:,0], samples[:,1], cmap=cm.get_cmap("tab10"), c=roll)
# plt.scatter(samples[:,0], samples[:,1], cmap=cm.get_cmap("tab10"), c=pitch)
# plt.colorbar()
plt.show()



# for size in range(8,12):
#     part = samples[sizes == size]
#     plt.scatter(part[:,0], part[:,1], cmap=cm.get_cmap("hot"))
# plt.show()

# https://matplotlib.org/examples/pylab_examples/custom_cmap.html