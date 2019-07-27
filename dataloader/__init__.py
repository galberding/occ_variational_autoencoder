
def get_dataset(mode, return_idx=False):
    ''' Returns the dataset.

        Args:
            model (nn.Module): the model which is used
            cfg (dict): config dictionary
            return_idx (bool): whether to include an ID field
        '''
    # Get split
    # TODO: adjust paths to datasets
    splits = {
        'train': "Path/to/set",
        'val': "Path/to/set",
        'test': "Path/to/set",
    }