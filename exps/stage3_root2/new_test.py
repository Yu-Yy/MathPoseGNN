import sys
import os
import argparse
import json
import cv2
from numpy.lib.function_base import extract
from scipy import optimize
from tqdm import tqdm

from torch.utils.data import DataLoader
# debug the file error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from lib.utils.post_3d import back_to_global
from cvpack.utils.logger import get_logger
from model.smap import SMAP
from model.pose2d_gnn import Pose_GCN
from model.refinenet import RefineNet
from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process
from exps.stage3_root2.test_util import *
from exps.stage3_root2.sync_group import pc_fused, pc_fused_connection
from dataset.custom_dataset import CustomDataset
from config import cfg
# import datalib
import dapalib
import cv2
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use("Agg")
from mpl_toolkits import mplot3d
import pickle
import torch.nn.functional as F
import time
from scipy.optimize import linear_sum_assignment # for gt matching

from model.gnn_refinenet import GNN_RefinedNet

dir_path = '/Extra/panzhiyu/CMU_data/gnn_testhm_r'

files = os.listdir(dir_path)

device = torch.device('cuda')

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

def off_the_result(poserefinenet_file):
    poserefine = Pose_GCN()
    poserefine.to(device) 
    state_dict = torch.load(poserefinenet_file, map_location=lambda storage, loc: storage)
    state_dict = state_dict['model']
    poserefine.load_state_dict(state_dict)
    batch_size = 4
    mpjpe = AverageMeter()
    mpjpe_refined = AverageMeter()
    precision = FPAverageMeter()
    recall = FPAverageMeter()
    precision_r = FPAverageMeter()
    recall_r = FPAverageMeter()
    for file in tqdm(files):
        with open(os.path.join(dir_path,file),'rb') as f:
            gnn_pair = pickle.load(f)
    
        matched_pred_single = gnn_pair['pred_single']
        matched_pred3d = gnn_pair['pred_3d']
        gt_3d = gnn_pair['gt_3d'] # cpu
        gt_bodys_2d = gnn_pair['gt_2d']
        cam_info = gnn_pair['cam']
        for v in matched_pred_single.keys():
            matched_pred_single[v] = matched_pred_single[v].reshape(-1,cfg.DATASET.MAX_PEOPLE,cfg.DATASET.KEYPOINT.NUM,7)
            matched_pred_single[v] = matched_pred_single[v].to(device)
            gt_bodys_2d[v] = gt_bodys_2d[v].to(device)
            gt_bodys_2d[v] = gt_bodys_2d[v].reshape(-1,cfg.DATASET.MAX_PEOPLE,cfg.DATASET.KEYPOINT.NUM,3)
            for attr in cam_info[v].keys():
                cam_info[v][attr] = cam_info[v][attr][0,...].to(device) # just one batch
        matched_pred3d = matched_pred3d.to(device)
        matched_pred3d = matched_pred3d.reshape(-1,cfg.DATASET.MAX_PEOPLE, cfg.DATASET.KEYPOINT.NUM,4)
        final_pred3d, residue_loss, refined_single_pos = poserefine(matched_pred_single, matched_pred3d)
        final_pred3d = torch.cat([final_pred3d, matched_pred3d[...,3:4]], dim=-1)
        for b in range(batch_size):
            pred_3d_b = matched_pred3d[b,...]
            gt_3d_b = gt_3d[b,...]
            final_pred3d_b = final_pred3d[b,...]
            pred_num = torch.sum(pred_3d_b[:,0,3]>0)
            gt_num = torch.sum(gt_3d_b[:,0,3] > 0)
            pred_3d_b = pred_3d_b[:pred_num,...]
            gt_3d_b = gt_3d_b[:gt_num,...]
            final_pred3d_b = final_pred3d_b[:pred_num,...]
            _,_ = val_match(pred_3d_b.cpu(),gt_3d_b,mpjpe, precision, recall, ap_threshold=2.5)
            _,_ = val_match(final_pred3d_b.cpu(), gt_3d_b, mpjpe_refined, precision_r, recall_r, ap_threshold=2.5)
    print(f'The orig MPJPE: {mpjpe.avg}, precision:{precision.avg}, recall: {recall.avg}')
    print(f'The refined MPJPE: {mpjpe_refined.avg}, prevision: {precision_r.avg}, recall: {recall_r.avg}')

def test_2d(hrnet_result_file, smap_result_file):
    with open(hrnet_result_file, 'rb') as f:
        hrnet_results = pickle.load(f)
    with open(smap_result_file,'rb') as f:
        smap_results = pickle.load(f)
        smap_results = smap_results['3d_pairs']
    # transverse the smap result
    smap_err = AverageMeter()
    hrnet_err = AverageMeter()
    precision_s = FPAverageMeter()
    recall_s = FPAverageMeter()
    precision_h = FPAverageMeter()
    recall_h = FPAverageMeter()
    dir_file = '/Extra/panzhiyu/CMU_data/'
    for smap_result in tqdm(smap_results):
        image_path = smap_result['image_path']
        img_path = image_path.replace(dir_file,'')
        if img_path in hrnet_results.keys():
            hrnet_result = hrnet_results[img_path]
        else:
            print(f'key err for {img_path}')
        smap_pred2d = smap_result['pred_aligned'] 
        smap_pred2d = smap_pred2d[:,:,[0,1,3]]
        gt_2d = smap_result['gt_2d']
        gt_2d = gt_2d[:,:,[0,1,3]]
        # validation
        # for the unified standard
        smap_pred2d = smap_pred2d[:,panoptic_to_unified,:] # 13 joints
        gt_2d = gt_2d[:,panoptic_to_unified,:]
        try:
            hrnet_result = hrnet_result[:,coco_to_unified,:]
        except:
            print(hrnet_result.shape)
            continue
        smap_pred2d = torch.from_numpy(smap_pred2d)
        gt_2d = torch.from_numpy(gt_2d)
        hrnet_result = torch.from_numpy(hrnet_result)
        _,_ = val_match2d(smap_pred2d, gt_2d, smap_err,precision_s, recall_s)
        _,_ = val_match2d(hrnet_result, gt_2d,hrnet_err,precision_h,recall_h)
    
    print(f'SMAP: err: {smap_err.avg}, precision@50:{precision_s.avg}, recall@50:{recall_s.avg}')
    print(f'Hrnet: err: {hrnet_err.avg}, precision@50:{precision_h.avg}, recall@50:{recall_h.avg}')





def val_match(pred_3d_total, gt_3d, mpjpe, precision, recall, threshold=50, ap_threshold=100): # FN
    # need to put in the CPU eval
    pred_3d = pred_3d_total[...,:3] # N X K X 3
    pred_vis = pred_3d_total[...,3] # N X K
    # pred_num = pred_3d.shape[0]
    gt_3d_position = gt_3d[...,:3]
    gt_vis = gt_3d[...,3]
    # actual_num = gt_3d.shape[0]
    mpjpe_save = dict()
    mpjpe_id = dict()
    for idx, (pose, pose_vis) in enumerate(zip(pred_3d, pred_vis)):
        temp_mpjpe = []
        for (gt, gt_valid) in zip(gt_3d_position, gt_vis):
            mask1 = (gt_valid > 0)
            mask2 = (pose_vis > 0)
            mask = mask1 * mask2
            if torch.sum(mask) == 0:
                temp_mpjpe.append(torch.tensor(5000))
                continue
            c_mpjpe = torch.mean(torch.norm((pose[mask,:] - gt[mask,:]), dim=-1))
            temp_mpjpe.append(c_mpjpe)
        min_gt = torch.argmin(torch.Tensor(temp_mpjpe))
        min_mpjpe = torch.min(torch.Tensor(temp_mpjpe))
        if min_mpjpe == 5000: # 
            continue
        if min_gt.item() in mpjpe_save.keys():
            if min_mpjpe < mpjpe_save[min_gt.item()]:
                mpjpe_save[min_gt.item()] = min_mpjpe
                mpjpe_id[min_gt.item()] = idx
        else:
            mpjpe_save[min_gt.item()] = min_mpjpe
            mpjpe_id[min_gt.item()] = idx
    error_list = torch.Tensor(list(mpjpe_save.values()))
    vis_label = torch.any(error_list > 40) # output the greater than 3
    mask_label = (error_list < threshold)
    filtered_id = list(mpjpe_id.values())

    ap_mask = (error_list < ap_threshold)
    tp_num = len(error_list[ap_mask])
    gt_num = gt_3d.shape[0]
    pred_num = pred_3d_total.shape[0]
    precision.update(tp_num, pred_num)
    recall.update(tp_num, gt_num)
    # if len(error_list[mask_label]) > 0:
    # error = torch.mean(error_list[mask_label]) # TODO: test the single view performance
    error = torch.mean(error_list) # TODO: test the single view performance
    # if torch.isnan(error):
    #     import pdb; pdb.set_trace() # check out one
    if ~torch.isnan(error):
        # print('nan') # using 500 instead
    # else:
        mpjpe.update(error.item())
    filtered_pose = pred_3d_total[filtered_id,...]
    filtered_pose = filtered_pose[mask_label,...]
    return vis_label, filtered_pose


def val_match2d(pred_2d_total, gt_2d, mpjpe, precision, recall, threshold=500, ap_threshold=50): # FN
    # need to put in the CPU eval
    pred_2d = pred_2d_total[...,:2] # N X K X 3
    pred_vis = pred_2d_total[...,2] # N X K
    # pred_num = pred_3d.shape[0]
    gt_2d_position = gt_2d[...,:2]
    gt_vis = gt_2d[...,2]
    # actual_num = gt_3d.shape[0]
    mpjpe_save = dict()
    mpjpe_id = dict()
    for idx, (pose, pose_vis) in enumerate(zip(pred_2d, pred_vis)):
        temp_mpjpe = []
        for (gt, gt_valid) in zip(gt_2d_position, gt_vis):
            mask1 = (gt_valid > 0)
            mask2 = (pose_vis > 0)
            mask = mask1 * mask2
            if torch.sum(mask) == 0:
                temp_mpjpe.append(torch.tensor(5000))
                continue
            c_mpjpe = torch.mean(torch.norm((pose[mask,:] - gt[mask,:]), dim=-1))
            temp_mpjpe.append(c_mpjpe)
        min_gt = torch.argmin(torch.Tensor(temp_mpjpe))
        min_mpjpe = torch.min(torch.Tensor(temp_mpjpe))
        if min_mpjpe == 5000: # 
            continue
        if min_gt.item() in mpjpe_save.keys():
            if min_mpjpe < mpjpe_save[min_gt.item()]:
                mpjpe_save[min_gt.item()] = min_mpjpe
                mpjpe_id[min_gt.item()] = idx
        else:
            mpjpe_save[min_gt.item()] = min_mpjpe
            mpjpe_id[min_gt.item()] = idx
    error_list = torch.Tensor(list(mpjpe_save.values()))
    vis_label = torch.any(error_list > 100) # output the greater than 3
    mask_label = (error_list < threshold)
    filtered_id = list(mpjpe_id.values())

    ap_mask = (error_list < ap_threshold)
    tp_num = len(error_list[ap_mask])
    gt_num = gt_2d.shape[0]
    pred_num = pred_2d_total.shape[0]
    precision.update(tp_num, pred_num)
    recall.update(tp_num, gt_num)
    # if len(error_list[mask_label]) > 0:
    # error = torch.mean(error_list[mask_label]) # TODO: test the single view performance
    error = torch.mean(error_list) # TODO: test the single view performance
    # if torch.isnan(error):
    #     import pdb; pdb.set_trace() # check out one
    if ~torch.isnan(error):
        # print('nan') # using 500 instead
    # else:
        mpjpe.update(error.item())
    filtered_pose = pred_2d_total[filtered_id,...]
    filtered_pose = filtered_pose[mask_label,...]
    return vis_label, filtered_pose


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FPAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    poserefine_file = '/home/panzhiyu/project/3d_pose/SMAP/model_logs_1020_singletest/stage3_root2/best_model.pth'
    off_the_result(poserefine_file)
    # hrnet_file = '/home/panzhiyu/project/3d_pose/SMAP/keypoints_validation_results.pkl'
    # smap_file = os.path.join('/home/panzhiyu/project/3d_pose/SMAP/model_logs_0821/stage3_root2/validation_result/','stage3_root2_generate_result_test_orig.pkl')
    # test_2d(hrnet_file, smap_file)
        

