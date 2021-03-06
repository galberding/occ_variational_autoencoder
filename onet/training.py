import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from metrics import make_3d_grid, compute_iou
from torch.nn import functional as F
import os
from torch import distributions as dist
from VoxelView.main import cloud2voxel
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from scipy import stats
from vis_vae import set_subplot_colormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def bin_cross_entropy(logits, occ):
    gamma = 0.6
    # fp = -occ*torch.log(logits)
    # fn = - (1-occ)*torch.log(1.0 - logits)
    # return fp + fn
    return -gamma*occ*torch.log(logits)- (1-gamma)*(1-occ)*torch.log(1.0 - logits)



def gen_plot(cloud, voxel, cloud_pred, voxel_pred):
    """Create a pyplot plot and save to buffer."""
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(projection='3d'), figsize=(10, 10))
    axes[0, 0].set_aspect('equal')
    axes[0, 1].set_aspect('equal')
    axes[1, 0].set_aspect('equal')
    axes[1, 1].set_aspect('equal')
    # voxel = np.array(voxel, dtype=np.float)
    axes[0, 0].voxels(voxel, edgecolor="k")
    axes[0, 1].voxels(voxel_pred, edgecolor="k")
    axes[1, 0].scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2])
    axes[1, 1].scatter(cloud_pred[:, 0], cloud_pred[:, 1], cloud_pred[:, 2])
    # plt.show()
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    return fig


class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, val_loader):
        ''' Performs  visualization.
        '''
        eval_list = list()
        count = 0
        for data in tqdm(val_loader):
            eval_fig = self.vis(data)
            eval_list.append(eval_fig)
            count += 1
            if count == 3:
                break
        return eval_list

    def get_zs_and_attr(self, val_loader):
        zs = list()
        count = 0
        attr = list()
        yaw = list()
        pitch = list()
        roll = list()
        device = self.device
        for batch in tqdm(val_loader):
            with torch.no_grad():
                kwargs = {}
                p = batch.get('points').to(device)
                occ = batch.get('points.occ').to(device)
                inputs = batch.get('inputs', torch.empty(p.size(0), 0)).to(device)
                transl = batch.get('transl').cpu().numpy()
                yaw.append(batch["yaw"].cpu().numpy()[0])
                pitch.append(batch["pitch"].cpu().numpy()[0])
                roll.append(batch["roll"].cpu().numpy()[0])
                q_z = self.model.infer_z(p, occ, inputs, **kwargs)
                z = q_z.rsample()
            attr.append(transl[0])
            zs.append(z.cpu().numpy()[0])
        return np.array(zs), np.array(attr), yaw, pitch, roll

    def vis_latent_attributes(self, val_loader):
        return self.vis_attr(*self.get_zs_and_attr(val_loader))

    def calculate_pears(self, val_loader):
        return self.calculate_pearson(*self.get_zs_and_attr(val_loader))


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.4, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.cmap = LinearSegmentedColormap.from_list('mycmap', ['blue', 'white', 'blue'])

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # # Compute iou
        # batch_size = points.size(0)
        #
        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)
        # print(p_out.probs.shape)
        # print(p_out.probs.min())
        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou
        # print(occ_iou_hat_np)
        # print(points_iou.numpy()[0])
        # pred_points = (points_iou.numpy()[0][occ_iou_hat_np[0] == 1])
        # voxel_pred = cloud2voxel(pred_points,1,32)
        #
        # # print(inputs.shape)
        # print(occ_iou[0].shape)
        # occ_iou = occ_iou.numpy()[0]
        # org_points = points_iou.numpy()[0]
        # print("Org: ",org_points[occ_iou == 1].shape)
        # plot_buf = gen_plot(org_points[occ_iou == 1], inputs[0], pred_points, voxel_pred)
        # eval_dict['img_buf'] = plot_buf

        #
        # # Estimate voxel iou
        # if voxels_occ is not None:
        #     voxels_occ = voxels_occ.to(device)
        #     points_voxels = make_3d_grid(
        #         (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
        #     points_voxels = points_voxels.expand(
        #         batch_size, *points_voxels.size())
        #     points_voxels = points_voxels.to(device)
        #     with torch.no_grad():
        #         p_out = self.model(points_voxels, inputs,
        #                            sample=self.eval_sample, **kwargs)
        #
        #     voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
        #     occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
        #     iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()
        #
        #     eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def vis(self, data, samples=1):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        threshold = self.threshold
        self.model.eval()
        kwargs = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        pred_points = (points_iou.cpu().numpy()[0][occ_iou_hat_np[0] == 1])
        voxel_pred = cloud2voxel(pred_points, 1, 32)

        # print(inputs.shape)
        # print(occ_iou[0].shape)
        occ_iou = occ_iou.cpu().numpy()[0]
        org_points = points_iou.cpu().numpy()[0]
        # print("Org: ",org_points[occ_iou == 1].shape)
        return gen_plot(org_points[occ_iou == 1], cloud2voxel(org_points[occ_iou == 1], 1, 32), pred_points, voxel_pred)

    def vis_attr(self, zs, transl, yaw, pitch, roll):
        # TODO: only import from vis_vae.py

        if zs.shape[1] > 2:
            samples_pca = PCA(n_components=2).fit_transform(zs)
            # samples_tsne = TSNE(n_components=2).fit_transform(zs)
        elif zs.shape[1] == 1:
            print("Unsupported dim of latent space!")
            exit(0)

        # samples = TSNE(n_components=2).fit_transform(samples)
        # print(samples.shape)

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        set_subplot_colormap(axes[0, 0], samples_pca, transl[:, 0], title="X-Translation", cmap="bwr")
        set_subplot_colormap(axes[0, 1], samples_pca, transl[:, 1], title="Y-Translation", cmap="bwr")
        set_subplot_colormap(axes[0, 2], samples_pca, transl[:, 2], title="Z-Translation", cmap="bwr")

        set_subplot_colormap(axes[1, 0], samples_pca, yaw, title="Yaw", cmap=self.cmap)
        set_subplot_colormap(axes[1, 1], samples_pca, pitch, title="Pitch", cmap=self.cmap)
        set_subplot_colormap(axes[1, 2], samples_pca, roll, title="Roll", cmap=self.cmap)
        return [fig]

    def calculate_pearson(self, zs, attr, yaw, pitch, roll):

        tags = ['0 x transl', '1 y transl', '2 z transl']
        tags_ypr = ['3 yaw', '4 pitch', '5 roll']
        ypr = [yaw, pitch, roll]
        zs_pears = dict()
        print(zs.shape)
        for i in range(zs.shape[1]):
            zs_pears[i] = dict()
            for j, tag in enumerate(tags):
                z_tmp = zs[:, i]
                t = attr[:, j]
                zs_pears[i][tag] = stats.pearsonr(z_tmp, t)[0]
                zs_pears[i][tags_ypr[j]] = stats.pearsonr(z_tmp, ypr[j])[0]
        return zs_pears

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        # c = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, inputs, **kwargs)
        # print("Point: ", p.size()," Occupancy: ", occ[0])
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()
        c = torch.empty(128, 0)
        # General points
        pr = self.model.decode(p, z, c, **kwargs)

        loss_i = F.binary_cross_entropy_with_logits(pr.logits, occ, reduction='none')
        loss_i = loss_i.sum(-1).mean()
        loss = loss + loss_i

        #loss_in =  bin_cross_entropy(pr.probs, occ).sum(-1).mean()
        #loss = loss + loss_in

        return loss
