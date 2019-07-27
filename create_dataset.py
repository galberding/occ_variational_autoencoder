from VoxelView.create_dataset_with_occs import gen_dataset
import sys


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage:\n"
              "python create_dataset <qube|sphere|pen> <#samples>")
        exit(0)


    # gen_dataset("sphere", "dataset/sphere/")
    # gen_dataset("pen", "dataset/pen/")
    voxel_model = sys.argv[1]
    samples = int(sys.argv[2])
    print(voxel_model)
    if voxel_model in "qube":
        gen_dataset("qube", "dataset/qube/", samples)
    elif voxel_model in "sphere":
        gen_dataset("sphere", "dataset/sphere/", samples)
    elif voxel_model in "pen":
        gen_dataset("pen", "dataset/pen/", samples)
    else:
        print("Model not known!")