from dataloader.core import VoxelDataset

# TODO: Find a more clear way to set the path to the dataset!
def get_dataset(mode, return_idx=False, dataset_path = "data/dataset/qube"):
    ''' Returns the dataset.

        Args:
            mode (String): determain which set to use
            return_idx (bool): whether to include an ID field
        '''
    # Get split
    # TODO: adjust paths to datasets
    splits = {
        'train': dataset_path + "train",
        'val': dataset_path + "val", # not handled yet
        'test': dataset_path + "test",
    }

    return VoxelDataset(splits[mode], return_idx=return_idx)
