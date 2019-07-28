import torch
import torch.optim as optim
from checkpoints import CheckpointIO
from onet import OccupancyNetwork

OUT_DIR = "out/pen"

# Create torch device for GPU computing
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

occ_net = OccupancyNetwork(device=device)
optimizer = optim.Adam(occ_net.parameters(), lr=1e-4)
checkpoint_io = CheckpointIO(OUT_DIR, model=occ_net)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    print("No model found!")