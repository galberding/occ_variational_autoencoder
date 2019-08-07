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
import io


def gen_plot(cloud, voxel, cloud_pred, voxel_pred):
    """Create a pyplot plot and save to buffer."""
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(projection='3d'), figsize=(10, 10))
    axes[0, 0].set_aspect('equal')
    axes[0, 1].set_aspect('equal')
    axes[1, 0].set_aspect('equal')
    axes[1, 1].set_aspect('equal')
    voxel = np.array(voxel, dtype=np.float)
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

        for data in tqdm(val_loader):
            eval_fig = self.vis(data)
            eval_list.append(eval_fig)
        return eval_list

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
                 vis_dir=None, threshold=0.1, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

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
        pred_points = (points_iou.numpy()[0][occ_iou_hat_np[0] == 1])
        voxel_pred = cloud2voxel(pred_points,1,32)

        # print(inputs.shape)
        # print(occ_iou[0].shape)
        occ_iou = occ_iou.numpy()[0]
        org_points = points_iou.numpy()[0]
        # print("Org: ",org_points[occ_iou == 1].shape)
        return gen_plot(org_points[occ_iou == 1], inputs[0], pred_points, voxel_pred)


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

        c = (inputs)
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        # print("Point: ", p.size()," Occupancy: ", occ[0])
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = self.model.decode(p, z, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss
