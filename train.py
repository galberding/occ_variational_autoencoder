from onet import OccupancyNetwork
from onet.training import Trainer
import torch
from dataloader import get_dataset
from dataloader.core import collate_remove_none, worker_init_fn
from checkpoints import CheckpointIO
import torch.optim as optim
from tensorboardX import SummaryWriter
import os



if __name__ == '__main__':
    batch_size = 1
    CHECKPOINT_PATH = "model"
    OUT_DIR = "out/pen"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Create torch device for GPU computing
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    # TODO: Automate dataset creation / adapt paths
    # Load training data
    train_dataset = get_dataset("train")
    # Load validation data
    val_dataset = get_dataset("test")



    print(train_dataset.len)
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)



    # create the model
    occ_net = OccupancyNetwork(device=device)
    optimizer = optim.Adam(occ_net.parameters(), lr=1e-4)
    # nparameters = sum(p.numel() for p in occ_net.parameters())
    # print(nparameters)

    # Restore the model
    checkpoint_io = CheckpointIO(OUT_DIR, model=occ_net, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    # metric_val_best = load_dict.get(
    #     'loss_val_best', -model_selection_sign * np.inf)
    # TODO: Validation scores for the model

    # Write to tensorboard
    logger = SummaryWriter(os.path.join(OUT_DIR, 'logs'))
    trainer = Trainer(occ_net, optimizer, device=device)
    # while True:
    #     epoch_it += 1
    for batch in train_loader:
        print(batch)
        loss = trainer.train_step(batch)
        print(loss)