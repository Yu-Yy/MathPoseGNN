import sys
import os
import argparse
import json
import cv2
from numpy.lib.function_base import extract
from scipy import optimize
from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import DataLoader
# debug the file error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from cvpack.utils.logger import get_logger
from model.smap import SMAP
from model.refinenet import RefineNet
from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process
from exps.stage3_root2.test_util import *
from exps.stage3_root2.sync_group import pc_fused, pc_fused_connection
from dataset.custom_dataset import CustomDataset
from exps.stage3_root2.config import cfg
# import datalib
import dapalib
import cv2
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use("Agg")
# from mpl_toolkits import mplot3d
import pickle
import torch.nn.functional as F
import time
from scipy.optimize import linear_sum_assignment # for gt matching

from model.gnn_refinenet import GNN_RefinedNet

kp_align_num = 2048

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14}
key_name = list(JOINTS_DEF.keys())
pairs = [[0, 1], [0, 2], [0, 9], [9, 10], [10, 11],
         [0, 3], [3, 4], [4, 5], [2, 12], [12, 13], 
         [13, 14], [2, 6], [6, 7], [7, 8]]

colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']

# === common joints definition between panoptic and coco
panoptic_to_unified = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
coco_to_unified = [0, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]
unified_joints_def = {
    'nose': 0,
    'l-shoulder': 1,
    'l-elbow': 2,
    'l-wrist': 3,
    'l-hip': 4,
    'l-knee': 5,
    'l-ankle': 6,
    'r-shoulder': 7,
    'r-elbow': 8,
    'r-wrist': 9,
    'r-hip': 10,
    'r-knee': 11,
    'r-ankle': 12,
}
unified_bones_def = [
    [0, 1], [0, 7],  # head
    [1, 2], [2, 3],  # left arm
    [7, 8], [8, 9],  # right arm
    [1, 4], [7, 10],  # trunk
    [4, 5], [5, 6],  # left leg
    [10, 11], [11, 12],  # right leg
]

unified_bones_def14 = [
    [0, 1],
    [0, 2], [0, 8],  # head
    [2, 3], [3, 4],  # left arm
    [8, 9], [9, 10],  # right arm
    [2, 5], [8, 11],  # trunk
    [5, 6], [6, 7],  # left leg
    [11, 12], [12, 13],  # right leg
]

File_Name = '/Extra/panzhiyu/CMU_data/keypoints_validation_results.json'
OFF_FILE = '/home/panzhiyu/project/3d_pose/SMAP/keypoints_validation_results.pkl'
with open(OFF_FILE,'rb') as f:
    pred_2d_results = pickle.load(f)


class SMAP_GNN(nn.Module):
    def __init__(self,shift_size = 3, is_train = True,device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.gnn_net = GNN_RefinedNet(layers_num = 4 ,out_ch = 64, device = device, max_people = cfg.DATASET.MAX_PEOPLE)
        self.smap = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
        self.shift_size = shift_size
        self.istrain = is_train
    
    def forward(self, images, valids, labels, rdepth, cam_paras, joints3d_global, joints3d_local_mul, scales): #, optimizer_s, scheduler_s
        batch_size = images.shape[0]
        batch_pc = [[[] for _ in range(batch_size)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        indx_match = [[[] for _ in range(batch_size)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        select_2d_match = [[[] for _ in range(batch_size)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        # use the view match, tag the view label
        viewidx_match = [[[] for _ in range(batch_size)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        cam_info = dict()
        # pose_2d_collect = dict()
        pose_2d_collect = [dict() for _ in range(batch_size)]
        smap_feature = dict()
        loss_smap = 0
        # select three view to generate
        cam_view = len(cfg.DATASET.CAM) # TODO: It has 5 views 
        if self.istrain:
            select = torch.randperm(cam_view)[:3]
        else: # infer do not select the views
            select = list(range(cam_view))
        for idx_v ,node in enumerate(cfg.DATASET.CAM):
            if idx_v not in select: # rand select three views
                continue
            view = list(node)[1]
            imgs = images[:,idx_v,...]
            if valids is not None:
                val = valids[:,idx_v,...]
                lab = labels[:,idx_v,...]
                rdep = rdepth[:,idx_v,...]
            else:
                val = None
                lab = None
                rdep = None
            meta_data = joints3d_local_mul[:,idx_v,...]
            cam_p = cam_paras[0][idx_v] # the parameters are same
            Kd = cam_p['distCoef']
            if self.istrain:
                loss_dict, outputs_2d, outputs_3d, outputs_rd, output_feature = self.smap(imgs, val, lab, rdep)
                losses = loss_dict['total_loss']
                loss_smap = losses + loss_smap
            else:
                outputs_2d, outputs_3d, outputs_rd, output_feature = self.smap(imgs, val, lab, rdep)
            outputs_3d = outputs_3d.detach().cpu()
            outputs_rd = outputs_rd.detach().cpu()
            for i in range(len(imgs)):
                if meta_data is not None: # meta data is the joints3d_local
                    # remove person who was blocked
                    new_gt_bodys = []
                    annotation = meta_data[i].cpu().numpy()
                    scale = scales[i] # It is equal for the same dataset
                    for j in range(len(annotation)):
                        if annotation[j, cfg.DATASET.ROOT_IDX, 3] > 1:
                            new_gt_bodys.append(annotation[j])
                    gt_bodys = np.asarray(new_gt_bodys)
                    if len(gt_bodys) == 0:
                        continue

                    if len(gt_bodys[0][0]) < 11:
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 7]
                        scale['cx'] = scale['img_width']/2
                        scale['cy'] = scale['img_height']/2
                    else:
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 8]
                        scale['cx'] = gt_bodys[0, 0, 9]
                        scale['cy'] = gt_bodys[0, 0, 10]
                else:
                    gt_bodys = None
                    # use default values
                    scale = {k: scales[k][i].numpy() for k in scales}
                    scale['f_x'] = scale['img_width']
                    scale['f_y'] = scale['img_width']
                    scale['cx'] = scale['img_width']/2
                    scale['cy'] = scale['img_height']/2


                hmsIn = outputs_2d[i].contiguous()
                hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255
                hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127
                rDepth = outputs_rd[i][0]
                pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True) # depth-aware part association lib # 存在预测分数
                if len(pred_bodys_2d) > 0:
                    pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape 4
                    pred_bodys_2d = pred_bodys_2d.numpy()

                # Get the heatmap value
                pafs_3d = outputs_3d[i].numpy().transpose(1, 2, 0)
                root_d = outputs_rd[i][0].numpy()
                
                paf_3d_upsamp = cv2.resize(
                    pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST) # get the rela depth map
                root_d_upsamp = cv2.resize(
                    root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)

                # generate 3d prediction bodys
                pred_bodys_2d = register_pred(pred_bodys_2d, None) #gt_bodys # make all the pred be zero

                if len(pred_bodys_2d) == 0:
                    continue

                pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scale, idx_v) # device pred_bodys_2d with relative depth
                pred_bodys_3d, adjust_2d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scale, idx_v, Kd) # device
                
                # generate the pointcloud
                pred_bodys_2d = adjust_2d.copy()

                # delete part of the low quality prediction
                vis_label = (adjust_2d[:,:,3] > 1)
                judge_num = np.sum(vis_label, axis=-1)
                pred_valid = (judge_num >= 8)
                adjust_2d = adjust_2d[pred_valid,...] # valid condition

                generate_pc_connection(adjust_2d, cam_p, scale, batch_pc, indx_match, select_2d_match, viewidx_match, view, i, self.shift_size, root_idx=0) # matching error? adjust_2d
                pose_2d_collect[i][view] = torch.from_numpy(pred_bodys_2d[:,:,[0,1,3]]).to(self.device)
            for k, v in cam_p.items():
                    cam_p[k] = torch.from_numpy(v).to(self.device).to(torch.float)
                    
            cam_p['pro'] = cam_p['K'] @ torch.cat([cam_p['R'], cam_p['t']], dim=-1)
            cam_p['pro_inv'] = torch.pinverse(cam_p['pro'])
            
            cam_info[view] = cam_p #

            smap_feature[view] = output_feature #.detach() # detach the grad from the smap part (not end to end)
        
        for k in range(cfg.DATASET.KEYPOINT.NUM): # TODO: can simplified
            for b in range(batch_size):
                if len(batch_pc[k][b]) >= 1 : 
                    # batch_pc[k][b] = torch.cat(batch_pc[k][b], dim = 0)
                    batch_pc[k][b] = np.concatenate(batch_pc[k][b], axis = 0)
                    batch_pc[k][b] = torch.from_numpy(batch_pc[k][b]).to(self.device).unsqueeze(0)
                    indx_match[k][b] = np.concatenate(indx_match[k][b], axis = -1)
                    indx_match[k][b] = torch.from_numpy(indx_match[k][b]).to(self.device) # to_device
                    select_2d_match[k][b] = np.concatenate(select_2d_match[k][b], axis=0)
                    select_2d_match[k][b] = torch.from_numpy(select_2d_match[k][b]).to(self.device).unsqueeze(0)
                    viewidx_match[k][b] = np.concatenate(viewidx_match[k][b], axis = -1)
                    viewidx_match[k][b] = torch.from_numpy(viewidx_match[k][b]).to(self.device)
                # else:
                    # print('Existing one batch contains no valid points')

        pred_bodys_3d = pc_fused_connection(batch_pc, indx_match, select_2d_match, viewidx_match, self.shift_size,cam_info, pose_2d_collect, self.device, root_idx=0)
        pred_bodys_3d = pred_bodys_3d.to(self.device)

        # if torch.sum((pred_bodys_3d[:,:,0,3] > 0)) == 0: # no valid pred 3d
        #     losses_t = dict()
        #     losses_t['smap_part'] = loss_smap
        #     losses_t['gnn_part'] = 0
        #     losses_t['total'] = loss_smap
        #     return losses_t
        matched_pred, matched_gt = matchgt(pred_bodys_3d.detach(), joints3d_global)
        pred_bodys_2d = project_views(matched_pred, cam_info, scale) # scale is equal for the same dataset

        _, refine_res = self.gnn_net(pred_bodys_2d, smap_feature, cam_info)
        refined_pose3d = torch.cat([matched_pred[:,:,:,:3] + refine_res, matched_pred[:,:,:,3:4]], dim = -1)
        # caluculate the loss
        pred_count = torch.sum(refined_pose3d[:,:,0,3],dim=-1, keepdim=True )
        valid_label = refined_pose3d[:,:,:,3:4] * (matched_gt[:,:,:,3:4] > 0.1)
        loss_temp = torch.norm((refined_pose3d[:,:,:,:3] - matched_gt[:,:,:,:3]) * valid_label, dim=-1) ** 2
        loss_m1 = torch.mean(loss_temp, dim=-1)         
        loss_3d = torch.mean(torch.sum(loss_m1 * (1/pred_count), dim = -1)) * 0.01 # 0.01 is the weight

        loss_total = loss_3d + loss_smap
        losses_t = dict()
        losses_t['smap_part'] = loss_smap
        losses_t['gnn_part'] = loss_3d
        losses_t['total'] = loss_total

        return losses_t, refined_pose3d, matched_gt


    def preload(self, pretrained_model_file):
        if os.path.isfile(pretrained_model_file):
            pretrained_state_dict = torch.load(pretrained_model_file) #,  map_location='cuda:0'
            model_pretrained = pretrained_state_dict['model']
            self.smap.load_state_dict(model_pretrained)
            del pretrained_state_dict
        else:
            print('Do not exist the pretrained file')

        
if __name__ == '__main__':
    pretrained_file = "/home/panzhiyu/project/3d_pose/SMAP/model_logs_0629/stage3_root2/best_model.pth"
    device = torch.device('cuda')
    new_model = SMAP_GNN()
    new_model = new_model.to(device)
    
    new_model.preload(pretrained_file)

