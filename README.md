# Occupancy Variational Autoencoder
Variational Autoencoder, based on [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks).

The designed encoder part of the VAE ca process input in form of voxel data. The encoder/network creates a 
latent representation of the input data. The encoder part is designed to take this representation, as well as points
in R^3 and predicts a occupancy probability for those points, if they are inside of the object (represented by the voxeldata).

## Data Generation

In order to generate some toy data to test the capabilities of the network I use [VoxelView](https://github.com/galberding/VoxelView/tree/217b4dc7073696ba147d196c504f8062bb207936)
to generate primative models (spheres, cubes, pens) in different size, rotation and translation.

![](assets/vox_sphere_small.png)
![](assets/vox_qube_small.png)
![](assets/vox_pen_small.png)

## Distribution in Latent Space

The first attempt was to train the network with one of the above presented model variations. 
After training the model should predict the latent representation of the samples from the test set.
The distribution of those samples is shown below.

![](assets/Latent_visualization_pens.png)

The pixelated lines in the plot are actually the voxeldata, downscaled and all captured by a fixed orientation.
Nevertheless, the plot shows a clear distribution where pens with similar orientation are close together.

The second attempt wasn't this successful. The network was supposed to learn the sphere model:

![](assets/Latent_visualization_spheres.png)

Again the plot shows the different voxelrepresentations of the sphere. Because of a bug spheres with small sizes
will become cubes which is clearly visible on the left side of the plot. Representations which are round(isch) tend to be grouped 
close together. 