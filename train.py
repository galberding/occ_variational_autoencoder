from onet import OccupancyNetwork
import torch
# Create torch device for GPU computing
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")


# TODO: Load training data
# TODO: Load validation data
# TODO: create the model
# TODO: Restore the model
# TODO: Validation scores for the model

if __name__ == '__main__':

    occ_net = OccupancyNetwork(device=device)

    nparameters = sum(p.numel() for p in occ_net.parameters())
    print(nparameters)