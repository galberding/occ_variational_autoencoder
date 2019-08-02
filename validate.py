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

OUT_DIR = "out/pen"
current_home = ""

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

test_dataset = get_dataset("train", dataset_path=current_home + "data/dataset/pen/")

# Create the dataloader
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=4, shuffle=False,
    collate_fn=collate_remove_none,
    worker_init_fn=worker_init_fn)

samples = []
sizes = []
for batch in test_loader:
    # print(batch["inputs"])
    # break
    print(occ_net.infer_z(None, None, batch["inputs"]).sample([1]), " size: ", batch["size"].numpy()[0])
    sizes.append(batch["size"].numpy()[0])
    samples.append(occ_net.infer_z(None, None, batch["inputs"]).sample([1]).numpy()[0,0])
samples = (np.array(samples))
sizes = np.array(sizes)

print(sizes)

for size in range(8,12):
    part = samples[sizes == size]
    plt.scatter(part[:,0], part[:,1])
plt.show()

# https://matplotlib.org/examples/pylab_examples/custom_cmap.html