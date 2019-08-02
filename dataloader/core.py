import os
import logging
from torch.utils import data
import numpy as np
import yaml
import torch


class VoxelDataset(data.Dataset):
    '''

    Loading the voxelrepresentations of Spheres, Pens and Qubes.

    '''

    def __init__(self, dataset_folder, return_idx=False):

        print("Init Dataset")
        self.return_idx = return_idx
        self.sample_paths = []
        for root, dirs, files in os.walk(dataset_folder):
            # print(root)

            if len(files) == 0:
                continue
            # print(files[0])
            self.sample_paths.append(os.path.join(root, files[0]))
        self.len = len(self.sample_paths)
        self.sample_paths = sorted(self.sample_paths)

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.len

    def __getitem__(self, idx):
        ''' Returns the data point.

        Args:
            idx (int): ID of data point
        '''

        # idx = torch.tensor(idx)
        # print("Load Voxel Item")
        sample = np.load(self.sample_paths[idx])
        inputs = sample['voxel']
        points_iou_occ = sample['occ']
        points_iou = sample['points']
        size = sample["size"]
        ypr = sample["yaw_pitch_roll"]
        yaw = ypr[0]
        pitch = ypr[1]
        roll = ypr[2]

        combinded = np.concatenate((points_iou, points_iou_occ[np.newaxis].T), axis=1)

        batch_samples = np.random.choice(combinded.shape[0], 2028)
        batch = combinded[batch_samples]
        occ = batch[:, -1]
        points = batch[:, :-1]
        inputs = np.array(inputs, dtype=np.float)

        # print("occ: ", occ.shape, " points: ", points.shape)
        points_iou = torch.tensor(points_iou, dtype=torch.float32)
        points_iou_occ = torch.tensor(points_iou_occ, dtype=torch.float32)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        # print(inputs)
        points = torch.tensor(points, dtype=torch.float32)
        occ = torch.tensor(occ, dtype=torch.float32)

        data = {
            'inputs': inputs,
            'points': points,
            'points.occ': occ,
            'points_iou': points_iou,
            'points_iou.occ': points_iou_occ,
            'idx': idx,
            'size': size,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }

        return data


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    # print(batch[0].shape)
    # print(batch)
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
