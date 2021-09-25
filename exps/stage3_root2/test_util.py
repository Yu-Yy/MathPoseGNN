"""
@author: Jianan Zhen
@contact: zhenjianan@sensetime.com
"""
import numpy as np
import cvxpy as cp
import copy
from numpy.lib.function_base import append
import torch
import torch.optim as optim
import time

from torch.optim.optimizer import Optimizer

from config import cfg
from lib.utils.post_3d import get_3d_points, back_to_global, get_3d_points_torch, back_to_global_torch, projectjointsPoints, projectjointsPoints_cp, projectjointsPoints_torch
from exps.stage3_root2.pointnet2_pro.pointnet2_modules import PointnetSAModuleDebug
from scipy.optimize import linear_sum_assignment
from exps.stage3_root2.optimization_util import optimize_step

joint_to_limb_heatmap_relationship = cfg.DATASET.PAF.VECTOR
paf_z_coords_per_limb = list(range(cfg.DATASET.KEYPOINT.NUM))
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)


def register_pred(pred_bodys, gt_bodys, root_n=2):
    if len(pred_bodys) == 0:
        return np.asarray([])
    if gt_bodys is not None:
        root_gt = gt_bodys[:, root_n, :2]
        root_pd = pred_bodys[:, root_n, :2]
        distance_array = np.linalg.norm(root_gt[:, None, :] - root_pd[None, :, :], axis=2)
        corres = np.ones(len(gt_bodys), np.int) * -1
        occupied = np.zeros(len(pred_bodys), np.int)
        while np.min(distance_array) < 30:
            min_idx = np.where(distance_array == np.min(distance_array))
            for i in range(len(min_idx[0])):
                distance_array[min_idx[0][i]][min_idx[1][i]] = 50
                if corres[min_idx[0][i]] >= 0 or occupied[min_idx[1][i]]:
                    continue
                else:
                    corres[min_idx[0][i]] = min_idx[1][i]
                    occupied[min_idx[1][i]] = 1
        new_pred_bodys = np.zeros((len(gt_bodys), len(gt_bodys[0]), 4), np.float)
        for i in range(len(gt_bodys)):
            if corres[i] >= 0:
                new_pred_bodys[i] = pred_bodys[corres[i]]
    else:
        new_pred_bodys = pred_bodys[pred_bodys[:, root_n, 3] != 0]
    return new_pred_bodys


def chain_bones(pred_bodys, depth_v, i, depth_0=0, root_n=2):
    if root_n == 2:
        start_number = 2
        pred_bodys[i][2][2] = depth_0
        pred_bodys[i][0][2] = pred_bodys[i][2][2] - depth_v[i][1]  # because of the order of the paf vector
    else:
        start_number = 1
        pred_bodys[i][0][2] = depth_0
    pred_bodys[i][1][2] = pred_bodys[i][0][2] + depth_v[i][0]  # paf vector
    for k in range(start_number, NUM_LIMBS):
        src_k = joint_to_limb_heatmap_relationship[k][0]
        dst_k = joint_to_limb_heatmap_relationship[k][1]
        pred_bodys[i][dst_k][2] = pred_bodys[i][src_k][2] + depth_v[i][k] # 直接相加 (get the depth value)


def generate_relZ(pred_bodys, paf_3d_upsamp, root_d_upsamp, scale, view_idx ,num_intermed_pts=10, root_n=2): # mid_hip
    # import pdb; pdb.set_trace()
    limb_intermed_coords = np.empty((2, num_intermed_pts), dtype=np.intp)
    depth_v = np.zeros((len(pred_bodys), NUM_LIMBS), dtype=np.float)
    depth_roots_pred = np.zeros(len(pred_bodys), dtype=np.float)
    for i, pred_body in enumerate(pred_bodys):
        if pred_body[root_n][3] > 0: # visible 系数很关键
            depth_roots_pred[i] = root_d_upsamp[int(pred_body[root_n][1]), int(pred_body[root_n][0])] * scale['scale'][view_idx] * scale['f_x'] # CAMPUS and SHELF need to change
            for k, bone in enumerate(joint_to_limb_heatmap_relationship):
                joint_src = pred_body[bone[0]]
                joint_dst = pred_body[bone[1]]
                if joint_dst[3] > 0 and joint_src[3] > 0: 
                    depth_idx = paf_z_coords_per_limb[k]
                    # Linearly distribute num_intermed_pts points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_intermed_pts))
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_intermed_pts))  # Same for the y coordinate
                    intermed_paf = paf_3d_upsamp[limb_intermed_coords[0, :],
                                                 limb_intermed_coords[1, :], depth_idx] # joint 坐标在 paf_z 的采样
                    min_val, max_val = np.percentile(intermed_paf, [10, 90])
                    intermed_paf[intermed_paf < min_val] = min_val
                    intermed_paf[intermed_paf > max_val] = max_val
                    mean_val = np.mean(intermed_paf) # 求预测
                    depth_v[i][k] = mean_val # 均值作为连接深度的值
            chain_bones(pred_bodys, depth_v, i, depth_0=0) # connect the depth
    return depth_roots_pred

def generate_relZ_torch(pred_bodys, paf_3d_upsamp, root_d_upsamp, scale, view_idx ,device,num_intermed_pts=10, root_n=2):
    # import pdb; pdb.set_trace()
    limb_intermed_coords = torch.empty((2, num_intermed_pts), dtype=torch.int).to(device)
    depth_v = torch.zeros((len(pred_bodys), NUM_LIMBS), dtype=torch.float).to(device)
    depth_roots_pred = torch.zeros(len(pred_bodys), dtype=torch.float).to(device)
    for i, pred_body in enumerate(pred_bodys):
        if pred_body[root_n][3] > 0: # visible 系数很关键
            depth_roots_pred[i] = root_d_upsamp[int(pred_body[root_n][1]), int(pred_body[root_n][0])] * scale['scale'][view_idx] * scale['f_x'] # reconstruct
            for k, bone in enumerate(joint_to_limb_heatmap_relationship):
                joint_src = pred_body[bone[0]]
                joint_dst = pred_body[bone[1]]
                if joint_dst[3] > 0 and joint_src[3] > 0: 
                    depth_idx = paf_z_coords_per_limb[k]
                    # Linearly distribute num_intermed_pts points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    limb_intermed_coords[1, :] = torch.round(torch.linspace(
                        joint_src[0], joint_dst[0], num_intermed_pts))
                    limb_intermed_coords[0, :] = torch.round(torch.linspace(
                        joint_src[1], joint_dst[1], num_intermed_pts))  # Same for the y coordinate
                    intermed_paf = paf_3d_upsamp[depth_idx, limb_intermed_coords[0, :].long(),
                                                 limb_intermed_coords[1, :].long()] # joint 坐标在 paf_z 的采样

                    min_val, max_val = torch.quantile(intermed_paf, torch.tensor([0.1,0.9]).to(device))
                    intermed_paf = torch.clamp(intermed_paf, min_val, max_val)
                    mean_val = torch.mean(intermed_paf) # 求预测
                    depth_v[i][k] = mean_val # 均值作为连接深度的值
            chain_bones(pred_bodys, depth_v, i, depth_0=0) # connect the depth
    return depth_roots_pred

def gen_3d_pose(pred_bodys, depth_necks, scale, view_idx, Kd):
    bodys = copy.deepcopy(pred_bodys)
    bodys[:, :, 0] = bodys[:, :, 0]/scale['scale'][view_idx] - (scale['net_width']/scale['scale'][view_idx]-scale['img_width'])/2
    bodys[:, :, 1] = bodys[:, :, 1]/scale['scale'][view_idx] - (scale['net_height']/scale['scale'][view_idx]-scale['img_height'])/2 # scale without distortion

    # 2D 变换为 原img 坐标系下的坐标表示
    # using bodys 与 depth neck generate the pointcloud
    K = np.asarray([[scale['f_x'], 0, scale['cx']], [0, scale['f_y'], scale['cy']], [0, 0, 1]])
    bodys_3d = get_3d_points(bodys, depth_necks, K, Kd) # generate the depth map
    for i in range(bodys_3d.shape[0]):
        for j in range(bodys_3d.shape[1]):
            if bodys_3d[i, j, 3] == 0:
                bodys_3d[i, j] = 0
    return bodys_3d, bodys

def gen_3d_pose_torch(pred_bodys, depth_necks, scale, view_idx, Kd, device):
    bodys = copy.deepcopy(pred_bodys)
    bodys[:, :, 0] = bodys[:, :, 0]/scale['scale'][view_idx] - (scale['net_width']/scale['scale'][view_idx]-scale['img_width'])/2
    bodys[:, :, 1] = bodys[:, :, 1]/scale['scale'][view_idx] - (scale['net_height']/scale['scale'][view_idx]-scale['img_height'])/2

    # import pdb; pdb.set_trace()
    # 2D 变换为 原img 坐标系下的坐标表示
    # using bodys 与 depth neck generate the pointcloud
    K = torch.tensor([[scale['f_x'], 0, scale['cx']], [0, scale['f_y'], scale['cy']], [0, 0, 1]]).to(device)
    Kd = torch.from_numpy(Kd).to(device)
    bodys_3d = get_3d_points_torch(bodys, depth_necks, K, Kd, device) # generate the depth map
    for i in range(bodys_3d.shape[0]):
        for j in range(bodys_3d.shape[1]):
            if bodys_3d[i, j, 3] == 0:
                bodys_3d[i, j] = 0
    return bodys_3d, bodys



def lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, device, root_n=2):
    root_3d_bodys = copy.deepcopy(pred_bodys_3d)
    root_2d_bodys = copy.deepcopy(pred_bodys_2d)
    score_after_refine = np.ones([pred_bodys_3d.shape[0], pred_bodys_3d.shape[1], 1], dtype=np.float)
    input_point = np.zeros((pred_bodys_3d.shape[0], 15, 5), dtype=np.float)
    input_point[:, root_n, :2] = root_2d_bodys[:, root_n, :2]
    input_point[:, root_n, 2:] = root_3d_bodys[:, root_n, :3]
    for i in range(len(root_3d_bodys)):
        if root_3d_bodys[i, root_n, 3] == 0:
            score_after_refine[i] = 0
        for j in range(len(root_3d_bodys[0])):
            if j != root_n and root_3d_bodys[i, j, 3] > 0:
                input_point[i, j, :2] = root_2d_bodys[i, j, :2] - root_2d_bodys[i, root_n, :2]
                input_point[i, j, 2:] = root_3d_bodys[i, j, :3] - root_3d_bodys[i, root_n, :3]
    input_point = np.resize(input_point, (input_point.shape[0], 75))
    inp = torch.from_numpy(input_point).float().to(device)
    pred = refine_model(inp)
    if pred.device.type == 'cuda':
        pred = pred.cpu().numpy()
    else:
        pred = pred.numpy()
    pred = np.resize(pred, (pred.shape[0], 15, 3))
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if j != root_n: #and pred_bodys_3d[i, j, 3] == 0:
                pred[i, j] += pred_bodys_3d[i, root_n, :3]
            else:
                pred[i, j] = pred_bodys_3d[i, j, :3]
    pred = np.concatenate([pred, score_after_refine], axis=2)
    return pred


def save_result_for_train_refine(pred_bodys_2d, pred_bodys_3d,gt_bodys, pred_rdepths,
                                 result, root_n=2):
    for i, pred_body in enumerate(pred_bodys_3d):
        if pred_body[root_n][3] != 0:
            pair = {}
            pair['pred_3d'] = pred_body.tolist()
            pair['pred_2d'] = pred_bodys_2d[i].tolist()
            pair['gt_3d'] = gt_bodys[i][:, 4:7].tolist()
            pair['root_d'] = pred_rdepths[i]
            result['3d_pairs'].append(pair)


def save_result(pred_bodys_2d, pred_bodys_3d, pred_g3d, aligned_g3d ,gt_bodys, gt_global,pred_rdepths, img_path,result): #

    pair = dict()
    pair['pred_2d'] = pred_bodys_2d.tolist()
    pair['pred_3d'] = pred_bodys_3d.tolist()
    pair['root_d'] = pred_rdepths.tolist()
    pair['pred_g3d'] = pred_g3d.tolist()
    pair['pred_aligned_g3d'] = aligned_g3d.tolist()
    pair['gt_g3d'] = gt_global.tolist()
    pair['image_path'] = img_path
    if gt_bodys is not None:
        pair['gt_3d'] = gt_bodys[:, :, 4:].tolist()
        pair['gt_2d'] = gt_bodys[:, :, :4].tolist()
    else:
        pair['gt_3d'] = list()
        pair['gt_2d'] = list()
    result['3d_pairs'].append(pair)

def save_result_multiview(pred_bodys_2d, pred_bodys_3d, pred_g3d, aligned_g3d ,gt_bodys, gt_global,pred_rdepths, img_path,result): #
    
    pair = dict()
    pair['pred_2d'] = pred_bodys_2d.tolist()
    pair['pred_3d'] = pred_bodys_3d.tolist()
    pair['root_d'] = pred_rdepths.tolist()
    pair['pred_g3d'] = pred_g3d.tolist()
    pair['pred_aligned_g3d'] = aligned_g3d.tolist()
    pair['gt_g3d'] = gt_global.tolist()
    pair['image_path'] = img_path
    if gt_bodys is not None:
        pair['gt_3d'] = gt_bodys[:, :, 4:].tolist()
        pair['gt_2d'] = gt_bodys[:, :, :4].tolist()
    else:
        pair['gt_3d'] = list()
        pair['gt_2d'] = list()
    result['3d_pairs'].append(pair)

def save_result_cmu(pred_bodys_2d, pred_aligned_2d, gt_2d, gt_bodys, img_path, cam, result): #
    
    pair = dict()
    pair['pred_2d'] = pred_bodys_2d #tolist()
    pair['pred_aligned'] = pred_aligned_2d #.tolist()
    pair['gt_2d'] = gt_2d #.tolist()
    pair['image_path'] = img_path
    pair['gt_3d'] = gt_bodys #.tolist()
    pair['cam'] = cam

    result['3d_pairs'].append(pair)

def generate_pc(pred_2d, cam_p, scale, batch_pc, batch_idx,sigma = 8, max_sampling = 100):
    # generate the pointcloud
    # depth_map = 5000 * np.ones((cfg.DATASET.KEYPOINT.NUM, scale['img_height'],scale['img_width']),dtype=np.float32)
    npose = len(pred_2d)
    # pc_collect = [[] for _ in range(cfg.DATASET.KEYPOINT.NUM)]  # collect according to the keypoint, same keypoint
    # generate according to the connection relationship
    for k in range(cfg.DATASET.KEYPOINT.NUM):
        extract_pose = pred_2d[:,k,:].copy() # Npose, 4
        # assign the kpoint
        for n in range(npose):
            tmp_size = sigma * extract_pose[n,3] # adjust the scope
            # if extract_pose[n,3] < 1:
            #     continue
            depth_val = extract_pose[n,2]
            mu_x = extract_pose[n,0]
            mu_y = extract_pose[n,1]
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if extract_pose[n,3] == 0 or ul[0] >= scale['img_width'] or ul[1] >= scale['img_height'] or br[0] < 0 or br[1] < 0:
                continue
            # get the image range
            # TODO: another choice, using heatmap to select
            img_x = max(0, ul[0]), min(br[0], scale['img_width'])
            img_y = max(0, ul[1]), min(br[1], scale['img_height'])
            # already get the scope, just unproject
            # get the meshgrid
            x_scope, y_scope = np.meshgrid(range(img_x[0], img_x[1]), range(img_y[0],img_y[1]))
            
            cor_2d = np.concatenate([x_scope.reshape(-1,1), y_scope.reshape(-1,1)], axis=1)
            pc = back_to_global(cor_2d, depth_val, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'])
            # generate the radiation direction
            pc_deep = back_to_global(cor_2d, depth_val + 10, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'])
            pc_direction = pc_deep - pc
            pc_direction = pc_direction / np.linalg.norm(pc_direction, axis=-1,keepdims=True)
            new_pc = np.concatenate([pc,pc_direction], axis = -1)
            # import pdb; pdb.set_trace()
            current_num = new_pc.shape[0]
            if current_num > max_sampling:
                # project
                sampling_idx = torch.randperm(current_num)[:max_sampling]
                new_pc = new_pc[sampling_idx,:]
            batch_pc[k][batch_idx].append(new_pc)
            # pc_collect[k].append(new_pc) # return N x 6 vector for one keypoint
            # depth_map[k,img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.where(depth_map[k,img_y[0]:img_y[1], img_x[0]:img_x[1]] > depth_val, depth_val,\
            #     depth_map[k,img_y[0]:img_y[1], img_x[0]:img_x[1]])
    # concate according to the keypoint
    # for k in range(cfg.DATASET.KEYPOINT.NUM):
    #     # pc_collect[k] = np.concatenate(pc_collect[k],axis=0)
    #     batch_pc[k][batch_idx] = np.concatenate(batch_pc[k][batch_idx],axis=0)
    # return pc_collect # return a list with length k

def generate_pc_torch(pred_2d, cam_p, scale, batch_pc, batch_idx, device, sigma = 8, max_sampling = 100): # update the logic
    # generate the pointcloud
    # depth_map = 5000 * np.ones((cfg.DATASET.KEYPOINT.NUM, scale['img_height'],scale['img_width']),dtype=np.float32)
    npose = len(pred_2d)
    # pc_collect = [[] for _ in range(cfg.DATASET.KEYPOINT.NUM)]  # collect according to the keypoint, same keypoint
    for k in range(cfg.DATASET.KEYPOINT.NUM):
        extract_pose = pred_2d[:,k,:].clone() # Npose, 4
        cor_3d_c = []
        # assign the kpoint
        for n in range(npose):
            tmp_size = sigma * extract_pose[n,3] # adjust the scope
            if extract_pose[n,3] < 1:
                continue
            # depth_val = extract_pose[n,2]
            depth_val = extract_pose[n:n+1,2:3]
            mu_x = extract_pose[n,0]
            mu_y = extract_pose[n,1]
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if extract_pose[n,3] == 0 or ul[0] >= scale['img_width'] or ul[1] >= scale['img_height'] or br[0] < 0 or br[1] < 0:
                continue
            # get the image range
            # TODO: another choice, using heatmap to select
            img_x = max(0, ul[0]), min(br[0], scale['img_width'])
            img_y = max(0, ul[1]), min(br[1], scale['img_height'])
            # already get the scope, just unproject
            # get the meshgrid
            x_scope, y_scope = torch.meshgrid(torch.arange(img_x[0], img_x[1]), torch.arange(img_y[0],img_y[1]))
            x_scope = x_scope.t().to(device) 
            y_scope = y_scope.t().to(device)

            # cor_2d = np.concatenate([x_scope.reshape(-1,1), y_scope.reshape(-1,1)], axis=1)
            cor_2d = torch.cat([x_scope.reshape(-1,1), y_scope.reshape(-1,1)], dim=1)
            current_num = cor_2d.shape[0]
            cor_3d = torch.cat([cor_2d, depth_val.repeat(current_num,1)], dim=1)
            if current_num > max_sampling:
                # project
                sampling_idx = torch.randperm(current_num)[:max_sampling]
                cor_3d = cor_3d[sampling_idx,:]
            cor_3d_c.append(cor_3d)

        cor_3d_c = torch.cat(cor_3d_c, dim=0)
        pc = back_to_global_torch(cor_3d_c, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'], device)
        cor_3d_c[:,2] = cor_3d_c[:,2] + 10
        pc_deep = back_to_global_torch(cor_3d_c, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'], device)
        pc_direction = pc_deep - pc
        pc_direction = pc_direction / torch.norm(pc_direction, dim=-1,keepdim=True)
        new_pc = torch.cat([pc,pc_direction], dim = -1)
        batch_pc[k][batch_idx].append(new_pc)
            # # optimize the logic
            # pc = back_to_global_torch(cor_2d, depth_val, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'], device)
            # # generate the radiation direction
            # pc_deep = back_to_global_torch(cor_2d, depth_val + 10, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'], device)
            # pc_direction = pc_deep - pc
            # # pc_direction = pc_direction / np.linalg.norm(pc_direction, axis=-1,keepdims=True)
            # pc_direction = pc_direction / torch.norm(pc_direction, dim=-1,keepdim=True)
            # new_pc = torch.cat([pc,pc_direction], dim = -1)
            # # import pdb; pdb.set_trace()
            # current_num = new_pc.shape[0]
            # if current_num > max_sampling:
            #     # project
            #     sampling_idx = torch.randperm(current_num)[:max_sampling]
            #     new_pc = new_pc[sampling_idx,:]
            # batch_pc[k][batch_idx].append(new_pc)


def generate_pc_connection(pred_2d, cam_p, scale, batch_pc, indx_match, select_2d_match, viewidx_match, view, batch_idx, tmp_size = 3 ,sigma = 8, root_idx=0):
    # generate the pointcloud
    npose = len(pred_2d)
    # varify all the npose first
    for n in range(npose):
        extract_pose = pred_2d[n,...].copy() # process the people in person tag
        if extract_pose[root_idx,3] == 0:
            continue
        for k in range(cfg.DATASET.KEYPOINT.NUM): #cfg.DATASET.KEYPOINT.NUM
            depth_val = extract_pose[k,2]
            mu_x = extract_pose[k,0]
            mu_y = extract_pose[k,1]
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size), int(mu_y + tmp_size)]
            unit_pc_num = (tmp_size*2)**2
            if extract_pose[k,3] <= 0.3 or ul[0] >= scale['img_width'] or ul[1] >= scale['img_height'] or br[0] < 0 or br[1] < 0: 
                invalid_idx = -1 * np.ones((1,(tmp_size*2)**2)) # init for the batch dim
                indx_match[k][batch_idx].append(invalid_idx)
                continue
            img_x = max(0, ul[0]), min(br[0], scale['img_width'])
            img_y = max(0, ul[1]), min(br[1], scale['img_height'])
            # assure the alignment
            if len(range(img_x[0], img_x[1])) < 2 * tmp_size :
                if img_x[0] == 0:
                    img_x = (0,2*tmp_size)
                elif img_x[1] == scale['img_width']:
                    img_x = [scale['img_width'] -2*tmp_size , scale['img_width']]
            elif len(range(img_y[0], img_y[1])) < 2*tmp_size :
                if img_y[0] == 0:
                    img_y = [0,2*tmp_size]
                elif img_y[1] == scale['img_height']:
                    img_y = [scale['img_height'] -2*tmp_size , scale['img_height']]
            # generate the pc's
            x_scope, y_scope = np.meshgrid(range(img_x[0], img_x[1]), range(img_y[0],img_y[1]))
            cor_2d = np.concatenate([x_scope.reshape(-1,1), y_scope.reshape(-1,1)], axis=1)
            pc = back_to_global(cor_2d, depth_val, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'])
            # generate the radiation direction
            pc_deep = back_to_global(cor_2d, depth_val + 10, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'])
            pc_direction = pc_deep - pc
            pc_direction = pc_direction / np.linalg.norm(pc_direction, axis=-1,keepdims=True)
            new_pc = np.concatenate([pc,pc_direction], axis = -1)
            assert new_pc.shape[0] == (tmp_size*2)**2, 'size mismatch'

            # match 2d (get every pc's center)
            cor_2d_center = extract_pose[k:k+1,:2]
            cor_2d_center = np.repeat(cor_2d_center, unit_pc_num, axis=0)
            match_2d = back_to_global(cor_2d_center, depth_val, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'])
            match_2d_deep = back_to_global(cor_2d_center, depth_val+10, cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'])
            match_direction = match_2d_deep - match_2d
            match_direction = match_direction / np.linalg.norm(match_direction, axis=-1,keepdims=True)
            match_center = np.concatenate([match_2d, match_direction], axis=-1)
            assert match_center.shape[0] == (tmp_size*2)**2, 'size mismatch'

            idx_init = len(batch_pc[k][batch_idx]) * (tmp_size*2)**2
            valid_index = np.arange(idx_init, idx_init + (tmp_size*2)**2)
            valid_index = valid_index[None, ...]
            view_tag_index = np.ones((1, unit_pc_num)) * view
            viewidx_match[k][batch_idx].append(view_tag_index)
            indx_match[k][batch_idx].append(valid_index) # add indx
            batch_pc[k][batch_idx].append(new_pc)
            select_2d_match[k][batch_idx].append(match_center)



            


def probalistic(di_pointcloud):
    # di_pointcloud with B x N x 6
    mu = di_pointcloud[:,:,:3]
    va_direction = di_pointcloud[:,:,3:]
    orth1 = np.ones(va_direction.shape)
    orth1[:,:,2] = -(va_direction[:,:,0] + va_direction[:,:,1]) /  va_direction[:,:,2]
    orth1 = orth1 / np.linalg.norm(orth1,axis=-1,keepdims=True)
    orth2 = np.cross(va_direction, orth1)
    # get the covariance matrix
    p = 9 * np.einsum('bij,bik-> bijk', va_direction, va_direction) + 1 * np.einsum('bij,bik-> bijk', orth1, orth1) + 1 * np.einsum('bij,bik-> bijk', orth2, orth2)
    return mu, p

def group_probalistic(orig_xyz, group_inds, group_mask):
    # orig: B, N, 21
    # group: B,Npoint,M
    # 
    # using unique acoording to different 
    
    B, npoints, nsamples = group_inds.shape
    dim_n = orig_xyz.shape[-1]
    orig_xyz_cp = orig_xyz[:,None,:,:].repeat(1,npoints,1,1)
    group_inds_tools = group_inds[:,:,:,None].repeat(1,1,1,dim_n)
    extracted = torch.gather(orig_xyz_cp,dim=2,index=group_inds_tools.type(torch.int64))
    extracted_sigmai = extracted[:,:,:,12:].reshape(B,npoints,nsamples,3,3)
    extracted_mu = extracted[:,:,:,:3].clone().reshape(B,npoints,nsamples,3,1)
    group_sigmai = torch.sum(extracted[:,:,:,12:] * (group_mask[...,None].repeat(1,1,1,9)), dim=2).reshape(B,npoints,3,3)
    # get the det value
    # compare_det = torch.linalg.det(group_sigmai) # time costing
    group_sigma = torch.inverse(group_sigmai) # inverse is time costing
    group_mu = torch.einsum('bnji,bnik->bnjk',group_sigma, torch.sum(torch.einsum('bnmji,bnmik->bnmjk', extracted_sigmai, extracted_mu) * (group_mask[...,None,None].repeat(1,1,1,3,1)),dim=2)) # error
    group_mu = group_mu.squeeze(-1)
    group_data = torch.cat([group_mu[:,1:,:],group_sigma[:,1:,...].flatten(-2),group_sigmai[:,1:,...].flatten(-2)], dim=-1)
    return group_mu, group_sigma, group_data

def get_model_corners(batch_points,device=torch.device('cuda')):
    # batch_points in tensor
    batch_num = batch_points.shape[0]
    min_x, max_x = torch.min(batch_points[:,:,0],dim=1,keepdim=True)[0], torch.max(batch_points[:,:,0],dim=1,keepdim=True)[0]
    min_y, max_y = torch.min(batch_points[:,:,1],dim=1,keepdim=True)[0], torch.max(batch_points[:,:,1],dim=1,keepdim=True)[0]
    min_z, max_z = torch.min(batch_points[:,:,2],dim=1,keepdim=True)[0], torch.max(batch_points[:,:,2],dim=1,keepdim=True)[0]
    corners_3d = torch.cat([torch.cat([min_x, min_y, min_z],dim=1).unsqueeze(1),
                            torch.cat([min_x, min_y, max_z],dim=1).unsqueeze(1),
                            torch.cat([min_x, max_y, min_z],dim=1).unsqueeze(1), 
                            torch.cat([min_x, max_y, max_z],dim=1).unsqueeze(1),
                            torch.cat([max_x, min_y, min_z],dim=1).unsqueeze(1),
                            torch.cat([max_x, min_y, max_z],dim=1).unsqueeze(1),
                            torch.cat([max_x, max_y, min_z],dim=1).unsqueeze(1),
                            torch.cat([max_x, max_y, max_z],dim=1).unsqueeze(1)],
                            dim=1)
    center_point = torch.mean(corners_3d,dim=1,keepdim=True)
    sigma_unit = torch.eye(3)[None,None,...].repeat(batch_num,1,1,1).to(device).reshape(batch_num,1,9)
    center_data = torch.cat([center_point,sigma_unit,sigma_unit],dim=-1)
    cat_xyz = torch.cat([center_data, batch_points],dim=1)
    return cat_xyz

def pc_fusion(demo_extracted):
    # batch = 1 version TODO: need to revise the version
    mu, sigma = probalistic(demo_extracted[None,...])
    # import pdb; pdb.set_trace()
    sigma_v = np.linalg.inv(sigma)
    # generate the 21 dim vector
    sigma_f = sigma.reshape(1,-1,9)
    sigma_vf = sigma_v.reshape(1,-1,9)
    device = torch.device('cuda')

    xyz = np.concatenate([mu, sigma_f, sigma_vf], axis = -1)
    # in tensor
    xyz_tensor = torch.tensor(xyz, dtype=torch.float).to(device)
    # xyz_tensor = xyz_tensor[None,...]
    
    nsample_1 = 512
    PA_FPSTEST = PointnetSAModuleDebug(npoint=10+1,  # mainly for downsampling
                radius=13,
                nsample=nsample_1,
                mlp=[21, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True)
                
    # group 1
    cat_xyz = get_model_corners(xyz_tensor)
    inds, group_inds, debug_xyz1 = PA_FPSTEST(cat_xyz)
    mask = ((group_inds[:,:,0:1].repeat(1,1,nsample_1) - group_inds) !=0)
    mask[:,:,0] = True
    group_mu, group_sigma, xyz2  = group_probalistic(cat_xyz, group_inds, mask) # First point is omitted
    flag = torch.sum(mask,dim=-1)
    return group_mu[:,1:,:]
 
def PoseSolver(pred_centers, nms_masks, n_root = 0):  # connection using the neck pose
    pred_centers = pred_centers.cpu()
    nms_masks = nms_masks.cpu() 
    batch_size = pred_centers.shape[1]
    output_pose = []
    for b in range(batch_size): # connection according to the distance
        nposes = torch.sum(nms_masks[n_root, b, ...])
        pred_pose = torch.zeros((nposes, cfg.DATASET.KEYPOINT.NUM, 3))   # process in cpu
        extracted_centers = pred_centers[n_root, b,...]
        c_pose = extracted_centers[nms_masks[n_root,b,...],:]
        pred_pose[:,n_root,:] = c_pose
        for k in range(NUM_LIMBS):
            src_k = joint_to_limb_heatmap_relationship[k][0]
            dst_k = joint_to_limb_heatmap_relationship[k][1]
            # using source to assign the dst
            src_pose = pred_pose[:,src_k,:]
            current_num = src_pose.shape[0]
            extracted_centers = pred_centers[dst_k,b,...]
            dst_centers = extracted_centers[nms_masks[dst_k,b,...],:]
            candi_num = dst_centers.shape[0]
            c_src = src_pose[:,None,:].repeat(1,candi_num,1)
            c_dst = dst_centers[None,:,:].repeat(current_num,1,1)
            dist_matrix = torch.norm((c_src - c_dst),dim=-1) 
            row_ind,col_ind=linear_sum_assignment(dist_matrix)
            # match_idx = torch.argmin(dist_matrix,dim=-1)

            # TODO: just connect without any consideration
            
            c_pose = dst_centers[col_ind,:]
            pred_pose[row_ind,dst_k,:] = c_pose 
            pred_pose = pred_pose[row_ind,...]
        
        output_pose.append(pred_pose)

    return output_pose

# def align_2d(pred_bodys_3d, pose_2d_collect, cam_info, delta_pose, pred_num, kp_num):
#     pred_bodys_3d_pos = pred_bodys_3d[:,:3]
#     pred_bodys_3d_vis = (pred_bodys_3d[:,3] >0 )
#     refined_3d_pose = pred_bodys_3d_pos + delta_pose
#     align_2d_collect = []
#     err = 0
#     for per_2d, cam_p in zip(pose_2d_collect, cam_info): # tranverse the views
#         project_2d_p = projectjointsPoints(pred_bodys_3d_pos, cam_p['K'], cam_p['R'], cam_p['t'],cam_p['distCoef'])
#         project_refined_2d = projectjointsPoints_cp(refined_3d_pose, cam_p['K'], cam_p['R'], cam_p['t'],cam_p['distCoef'])
#         # project_refined_2d = project_refined_2d.reshape(pred_num, -1, 2)
#         # judge the visible joint
#         x_check = np.bitwise_and(project_2d_p[:, 0] >= 0, 
#                                     project_2d_p[:, 0] <= 1920 - 1) #(15,) bool
#         y_check = np.bitwise_and(project_2d_p[:, 1] >= 0,
#                                     project_2d_p[:, 1] <= 1080 - 1) # just fix the coord
#         check = np.bitwise_and(x_check, y_check) # N
#         # match the project_2d with the per_2d
#         er_match = dict()
#         er_id = dict()
#         for idx in range(pred_num):
#             project_2d = project_2d_p[idx*kp_num:(idx+1)*kp_num,...]
#             project_vis = pred_bodys_3d_vis[idx*kp_num:(idx+1)*kp_num] * check[idx*kp_num:(idx+1)*kp_num]
#             # K x 2 and K
#             temp_er = []
#             if np.sum(project_vis) < 5: # at least 5 points
#                 continue
#             for off_2d in per_2d:
#                 off_2d_pos = off_2d[:,:2]
#                 off_2d_vis = (off_2d[:,2] > 0.5)
#                 mask = project_vis * off_2d_vis
#                 if np.sum(mask) == 0:
#                     temp_er.append(1000)
#                     continue
#                 c_err = np.mean(np.linalg.norm(project_2d[mask,...] - off_2d_pos[mask,...], axis=-1))
#                 temp_er.append(c_err)
#             min_gt = np.argmin(temp_er)
#             min_er = np.min(temp_er)
#             if min_gt in er_match.keys():
#                 if min_er < er_match[min_gt]:
#                     er_match[min_gt] = min_er
#                     er_id[min_gt] = list(range(idx*kp_num, (idx+1)*kp_num)) # 15 is the kp num
#             else:
#                 er_match[min_gt] = min_er
#                 er_id[min_gt] = list(range(idx*kp_num, (idx+1)*kp_num))
#         project_idx = np.asarray(list(er_id.values()))
#         project_idx = project_idx.reshape(-1)
#         if len(project_idx) == 0:
#             align_2d_collect.append(None) # may exist the None option
#         match_idx = np.asarray(list(er_id.keys()))
#         # valid_label = np.array(list(er_match.values())) < t xhreshold 

#         c_pose = per_2d[match_idx,...]
#         c_refined_2d = project_refined_2d[project_idx,...]
#         check_new = check.reshape(-1)
#         check_c = check_new[project_idx]
#         vis_pose = (c_pose[...,2] > 0)
#         vis_pose = vis_pose.reshape(-1)
#         c_pose_pos = c_pose[...,:2].reshape(-1,2)
#         total_mask = check_c * vis_pose * pred_bodys_3d_vis[project_idx]
#         total_mask = total_mask.astype(np.int)
#         err = err + cp.sum(cp.multiply(cp.power(cp.norm(c_refined_2d - c_pose_pos, axis=1),2) , total_mask))   #np.sum(np.linalg.norm(c_refined_2d - c_pose, axis=-1) * total_mask) 
#         align_2d_collect.append(c_pose)
    
#     return err   

        

# def align_2d(pred_bodys_3d, pose_2d_collect, cam_info):
#     pred_bodys_3d_pos = pred_bodys_3d[:,:,:3]
#     pred_num = pred_bodys_3d_pos.shape[0]
#     kp_num = pred_bodys_3d_pos.shape[1]
#     pred_bodys_3d_vis = (pred_bodys_3d[:,:,3] > 0)
#     pred_bodys_3d_r = pred_bodys_3d_pos.reshape(-1,3)

#     align_2d_collect = []
#     project_id_collect = []
#     for per_2d, cam_p in zip(pose_2d_collect, cam_info):
#         project_2d = projectjointsPoints_torch(pred_bodys_3d_r, cam_p['K'], cam_p['R'], cam_p['t'],cam_p['distCoef'])
#         project_2d = project_2d.reshape(pred_num,kp_num,2)
#         x_check = torch.bitwise_and(project_2d[:, :, 0] >= 0, 
#                                     project_2d[:, :, 0] <= 1920 - 1) #(15,) bool
#         y_check = torch.bitwise_and(project_2d[:, :, 1] >= 0,
#                                     project_2d[:, :, 1] <= 1080 - 1) # just fix the coord
#         check = torch.bitwise_and(x_check, y_check) # N
#         er_match = dict()
#         er_id = dict()
#         for idx, (p_2d, p_check, vis_3d_p) in enumerate(zip(project_2d, check, pred_bodys_3d_vis)):
#             match_check = p_check * vis_3d_p # valid project2d and valid region
#             temp_er = []
#             if torch.sum(match_check) < 5:
#                 continue
#             for off_2d in per_2d:
#                 off_2d_pose = off_2d[:,:2]
#                 off_2d_vis = (off_2d[:,2] > 0.5)
#                 mask = match_check * off_2d_vis
#                 if torch.sum(mask) == 0:
#                     temp_er.append(1000)
#                     continue
#                 c_err = torch.mean(torch.norm(off_2d_pose[mask,...] - p_2d[mask,...], dim=-1))
#                 temp_er.append(c_err)
#             min_gt = torch.argmin(torch.tensor(temp_er))
#             min_er = torch.min(torch.tensor(temp_er))
#             if min_gt.item() in er_match.keys():
#                 if min_er < er_match[min_gt.item()]:
#                     er_match[min_gt.item()] = min_er
#                     er_id[min_gt.item()] = idx
#             else:
#                 er_match[min_gt.item()] = min_er
#                 er_id[min_gt.item()] = idx
        
#         project_idx = torch.tensor(list(er_id.values()))
#         match_idx = torch.tensor(list(er_id.keys()))
#         if len(match_idx) == 0:
#             align_2d_collect.append(None)
#             project_id_collect.append(None)
#             continue
#         c_pose = per_2d[match_idx,...]
#         check_new = check[match_idx,...]
#         c_pose[:,:,2] = c_pose[:,:,2] * check_new # fuse two judgement
#         align_2d_collect.append(c_pose)
#         project_id_collect.append(project_idx)

#         # c_refined_2d = project_refined_2d[project_idx,...] # 借用匹配结果
#         # check = check[project_idx,...]
#         # vis_pose = (c_pose[...,2] > 0.5)
#         # c_pose_pos = c_pose[:,:,:2]
#         # total_mask = check * vis_pose * pred_bodys_3d_vis[project_idx,...]
#         # loss = loss + torch.sum((torch.norm(c_refined_2d - c_pose_pos, dim=-1) ** 2) * total_mask)

#     return align_2d_collect, project_id_collect







def optimize_3d(pred_bodys_3d, pose_2d_collect, cam_info ,device):
    # pred_bodys_3d is N x K x 3
    # the cam info need to converted to torch
    
    # for cam_p in cam_info:
    #     for k, v in cam_p.items():
    #         cam_p[k] = torch.from_numpy(v).to(device).to(torch.float)

    # align_2d_pose, project_idx_collect = align_2d(pred_bodys_3d, pose_2d_collect, cam_info)
    pred_num = pred_bodys_3d.shape[0]
    kp_num = pred_bodys_3d.shape[1]
    reg_p = 0.01
    refine_tool = optimize_step(reg_p, pred_num, kp_num,  device).to(device) #pred_bodys_3d[:,:,:3],
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, refine_tool.parameters()), lr = 1e-1)
    max_iter = 70
    for i in range(max_iter):
        refined_3d, loss = refine_tool(pred_bodys_3d, pose_2d_collect, cam_info)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    return refined_3d


    # pred_num = pred_bodys_3d.shape[0]
    # kp_num = pred_bodys_3d.shape[1]
    # pred_bodys_3d = pred_bodys_3d.reshape(-1,4)
    # delta_pose = cp.Variable(((pred_num*kp_num), 3))
    # err_2d = align_2d(pred_bodys_3d, pose_2d_collect, cam_info, delta_pose,pred_num, kp_num) # V: N x k X 2
    # # find the corresponding 2d pose, and do the loss
    # regularization =  cp.norm(delta_pose)
    # total_loss = err_2d + regularization
    # prob = cp.Problem(cp.Minimize(total_loss))
    # prob.solve()


def project_views(pred_bodys_3d_r, cam_info, scale):
    # pred_bodys_3d (1,15,4)
    # pred_num, kp_num, _ = pred_bodys_3d_r.shape
    kp_num, _ = pred_bodys_3d_r.shape
    # pred_bodys_3d = pred_bodys_3d_r[:,:,:3]
    pred_bodys_3d = pred_bodys_3d_r[:,:3]
    # pred_bodys_3d = pred_bodys_3d.reshape(-1,3)
    pred_vis = pred_bodys_3d_r[:,3]
    pred_2d_project = dict()

    for v,cam_p in cam_info.items():
        project_2d = projectjointsPoints_torch(pred_bodys_3d, cam_p['K'], cam_p['R'], cam_p['t'],cam_p['distCoef'])
        # project_2d = project_2d.reshape(pred_num,kp_num,2)
        x_check = torch.bitwise_and(project_2d[:, 0] >= 0, 
                                    project_2d[:, 0] <= 1920 - 1) #(15,) bool
        y_check = torch.bitwise_and(project_2d[:, 1] >= 0,
                                    project_2d[ :, 1] <= 1080 - 1) # just fix the coord
        check = torch.bitwise_and(x_check, y_check) # N
        
        project_2d[:, 0] =(project_2d[:, 0] + (scale['net_width']/scale['scale'][0]-scale['img_width'])/2) * scale['scale'][0]
        project_2d[:, 1] =(project_2d[:, 1] + (scale['net_height']/scale['scale'][0]-scale['img_height'])/2) * scale['scale'][0] # different views has the same

        pred_2d_vis = pred_vis * check
        pred_2d = torch.cat([project_2d, pred_2d_vis[:,None]], dim = -1)
        pred_2d_project[v] = pred_2d

    return pred_2d_project