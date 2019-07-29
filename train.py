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
    batch_size = 5
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
    test_dataset = get_dataset("test")



    print(train_dataset.len)
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, num_workers=4, shuffle=False,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn)



    # create the model
    logger = SummaryWriter(os.path.join(OUT_DIR, 'logs'))
    occ_net = OccupancyNetwork(device=device, logger=logger)
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
    trainer = Trainer(occ_net, optimizer, device=device)
    epoch_it = 0
    # it = 0
    checkpoint_every = 100
    eval_network = 100
    while epoch_it < 1000:
        epoch_it += 1
        for batch in train_loader:
            it += 1
            # print(batch)
            loss = trainer.train_step(batch)
            logger.add_scalar('train/loss', loss, it)
            print("Epoch: ", epoch_it," Iteration: ", it, " Loss: ",loss)
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                print('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it)
            if (eval_network > 0 and (it % eval_network) == 0):
                print("Evaluate network")
                eval_dict = trainer.evaluate(test_loader)
                logger.add_scalar('val/loss', eval_dict['loss'],  it)
                logger.add_scalar('val/rec_error', eval_dict['rec_error'],  it)
                logger.add_scalar('val/kl-div', eval_dict['kl'],  it)
                # logger.add_scalar('val/iou', eval_dict['iou'],  it)
