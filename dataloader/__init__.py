from dataloader.core import VoxelDataset


def get_dataset(mode, return_idx=False):
    ''' Returns the dataset.

        Args:
            mode (String): determain which set to use
            return_idx (bool): whether to include an ID field
        '''
    # Get split
    # TODO: adjust paths to datasets
    splits = {
        'train': "Path/to/set",
        'val': "Path/to/set",
        'test': "Path/to/set",
    }

    return VoxelDataset(splits[mode], return_idx=return_idx)
