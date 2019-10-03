from VoxelView.create_dataset_with_occs import gen_dataset
import sys
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create train and test set for given modeltype.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m","--model", nargs=1, metavar="<pen|sphere|qube>", required=True, type=str)
    parser.add_argument("-s", "--samples", nargs=1, default=1000, type=int)

    # TODO: Add sizes, path and points to generate
    args = parser.parse_args()
    current_dir = (os.getcwd())
    dataset_path = "data/dataset"
    # dataset_path = "../generativeVAE/data/dataset"
    voxel_model = args.model[0]
    samples = args.samples[0]
    print(samples, voxel_model)

    if voxel_model not in ["qube", "sphere", "pen"]:
        print("Model not known!")
        exit(0)
    path = (os.path.join(current_dir, dataset_path, "qube", ''))
    print("pass!")
    # exit(1)
    gen_dataset(voxel_model, path, samples, pen_sizes=[9,9,9,9,9,9,9,9,9,9])
