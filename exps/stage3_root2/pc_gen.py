"""
@author: Jianan Zhen
@contact: zhenjianan@sensetime.com
"""

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

save_dir = '/Extra/panzhiyu/CMU_data/gnn_hm/'

with open(OFF_FILE,'rb') as f:
    pred_2d_results = pickle.load(f)

def generate_3d_point_pairs(model, refine_model, data_loader, cfg, logger, device, 
                            output_dir='', total_iter='infer'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    if refine_model is not None:
        refine_model.eval()

    gnn_net = GNN_RefinedNet(layers_num = 4 ,out_ch = 64, device = device, max_people = cfg.DATASET.MAX_PEOPLE)
    gnn_net = gnn_net.to(device)
    result = dict()
    result['model_pattern'] = cfg.DATASET.NAME
    result['3d_pairs'] = []
    # 3d_pairs has items like{'pred_2d':[[x,y,detZ,score]...], 'gt_2d':[[x,y,Z,visual_type]...],
    #                         'pred_3d':[[X,Y,Z,score]...], 'gt_3d':[[X,Y,X]...],
    #                         'root_d': (abs depth of root (float value) pred by network),
    #                         'image_path': relative image path}

    data = tqdm(data_loader) if is_main_process() else data_loader

    max_people = 10
    # create the val dict
    kp_set = [AverageMeter() for _ in range(cfg.DATASET.KEYPOINT.NUM)]
    registor = AverageMeter()
    mpjpe = AverageMeter()
    mpjpe_g3d = AverageMeter()
    kp_set_g3d = [AverageMeter() for _ in range(cfg.DATASET.KEYPOINT.NUM)]
    fp_total = FPAverageMeter()
    kp_fp = [FPAverageMeter() for _ in range(cfg.DATASET.KEYPOINT.NUM)]
    shift_size = 3
    root_dir = '/Extra/panzhiyu/CMU_data/'
    # cuda test
    # test = torch.rand((500,500)).to(device)
    # aa = torch.rand((500,)).to(device)
    # t = test[aa > 0.5]
    # save_idx = 0 # shut point is 2417
    save_idx = 0
    for idx, batch in enumerate(data):
        # if cfg.TEST_MODE == 'run_inference':
        #     imgs, img_path, scales = batch
        #     meta_data = None
        # else:
        # imgs, meta_data, img_path, scales = batch #
        # debug_count = 0
        imgs_mul, cam_paras, joints3d_global, joints3d_local_mul, scales = batch
        imgs_mul = imgs_mul.to(device)
        joints3d_global= joints3d_global.to(device)
        batch_size = imgs_mul.shape[0]
        # import pdb; pdb.set_trace()
        # json_file = os.path.join('/Extra/panzhiyu/CMU_data/cam_para.pkl')
        # with open(json_file, 'wb') as f:
        #     pickle.dump(cam_paras[0], f)

        # Select the test set  !!!!
        # c_image = scales[0]['img_paths'][0].split(root_dir)[-1]
        hm_collect = []
        # cam_c = []
        # pose2d_c = []
        # if c_image not in pred_2d_results.keys():
        #     continue
        cam_info = dict()
        # pose_2d_collect = dict()
        pose_2d_collect = [dict() for _ in range(batch_size)]
        smap_feature = dict()
        # Batch = 1 matching
        # try:
        #     img_path_p = os.path.join('/Extra/panzhiyu/CMU_data',pred_2d_results[IDX_PRED]['image_name'])
        #     this_img = scales[0]['img_paths'][0] # the first view
        #     if this_img != img_path_p:
        #         # debug_count = debug_count + 1
        #         # if debug_count > 100:
        #         #     assert False, 'mismatch error'
        #         continue
        #     else:
        #         while img_path_p in scales[0]['img_paths']:
        #             IDX_PRED = IDX_PRED + 1
        #             img_path_p = os.path.join('/Extra/panzhiyu/CMU_data',pred_2d_results[IDX_PRED]['image_name'])
        # except:
        #     # The end of the list
        #     logger.info('End of the list')
        #     break
        # match the vali_joints
        with torch.no_grad():
            # in multiple views
            batch_pc = [[[] for _ in range(batch_size)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
            indx_match = [[[] for _ in range(batch_size)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
            select_2d_match = [[[] for _ in range(batch_size)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
            # use the view match, tag the view label
            viewidx_match = [[[] for _ in range(batch_size)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
            for idx_v ,node in enumerate(cfg.DATASET.CAM):
                # process the node into the view int
                view = list(node)[1]
                imgs = imgs_mul[:,idx_v,...]
                meta_data = joints3d_local_mul[:,idx_v,...]
                cam_p = cam_paras[0][idx_v] # the parameters are same
                Kd = cam_p['distCoef'] 
                outputs_2d, outputs_3d, outputs_rd, output_feature = model(imgs) # the batch dim is 1
                outputs_3d = outputs_3d.cpu()
                outputs_rd = outputs_rd.cpu()
                hm_collect.append(outputs_2d.clone().cpu().unsqueeze(0)) # all save
                # if cfg.DO_FLIP:
                #     outputs_2d_flip, outputs_3d_flip, outputs_rd_flip = model(imgs_flip)
                #     outputs_2d_flip = torch.flip(outputs_2d_flip, dims=[-1])
                #     # outputs_3d_flip = torch.flip(outputs_3d_flip, dims=[-1])
                #     # outputs_rd_flip = torch.flip(outputs_rd_flip, dims=[-1])
                #     keypoint_pair = cfg.DATASET.KEYPOINT.FLIP_ORDER
                #     paf_pair = cfg.DATASET.PAF.FLIP_CHANNEL
                #     paf_abs_pair = [x+kpt_num for x in paf_pair]
                #     pair = keypoint_pair + paf_abs_pair
                #     for i in range(len(pair)):
                #         if i >= kpt_num and (i - kpt_num) % 2 == 0:
                #             outputs_2d[:, i] += outputs_2d_flip[:, pair[i]]*-1
                #         else:
                #             outputs_2d[:, i] += outputs_2d_flip[:, pair[i]]
                #     outputs_2d[:, kpt_num:] *= 0.5

                # paf_3d = F.interpolate(outputs_3d, size=(cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1])) # upsample the result
                # root_d = F.interpolate(outputs_rd, size=(cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1]))
                for i in range(len(imgs)):
                    if meta_data is not None: # meta data is the joints3d_local
                        # remove person who was blocked
                        new_gt_bodys = []
                        annotation = meta_data[i].numpy()
                        scale = scales[i] # It is equal for the same dataset
                        for j in range(len(annotation)):
                            if annotation[j, cfg.DATASET.ROOT_IDX, 3] > 1:
                                new_gt_bodys.append(annotation[j])
                        gt_bodys = np.asarray(new_gt_bodys)
                        if len(gt_bodys) == 0:
                            continue
                        # groundtruth:[person..[keypoints..[x, y, Z, score(0:None, 1:invisible, 2:visible), X, Y, Z,
                        #                                   f_x, f_y, cx, cy]]]
                        # changed into the tensor

                        # gt_bodys = torch.from_numpy(gt_bodys).to(device)

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
                    # if the first pair is [1, 0], uncomment the code below
                    # hmsIn[cfg.DATASET.KEYPOINT.NUM:cfg.DATASET.KEYPOINT.NUM+2] *= -1
                    # outputs_3d[i, 0]n *= -1
                    hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255
                    hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127
                    rDepth = outputs_rd[i][0]
                    # vis the rDepth


                    pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True) # depth-aware part association lib # 存在预测分数
                    
                    if len(pred_bodys_2d) > 0:
                        pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape 4
                        pred_bodys_2d = pred_bodys_2d.numpy()

                    pafs_3d = outputs_3d[i].numpy().transpose(1, 2, 0)
                    root_d = outputs_rd[i][0].numpy()


                    # paf_3d_upsamp = paf_3d[i]
                    # root_d_upsamp = root_d[i][0]
                    
                    paf_3d_upsamp = cv2.resize(
                        pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST) # get the rela depth map
                    root_d_upsamp = cv2.resize(
                        root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)


                    # generate 3d prediction bodys
                    pred_bodys_2d = register_pred(pred_bodys_2d, None) #gt_bodys # make all the pred be zero

                    # if len(pred_bodys_2d) == 0:
                    #     pred_bodys_2d = torch.tensor([])
                    # pred_bodys_2d = pred_bodys_2d[pred_bodys_2d[:, 2, 3] != 0]
                    # pred_bodys_2d = pred_bodys_2d.to(device)

                    # replace the 2D rped with the HRNET

                    # img_path_p = os.path.join('/Extra/panzhiyu/CMU_data',pred_2d_results[NEW_PRED]['image_name'])
                    # this_img = scale['img_paths'][idx_v]
                    # if img_path_p == this_img:
                    #     pred_hrnet_pose = []
                    #     while(img_path_p == this_img):
                    #         pred_pose = pred_2d_results[NEW_PRED]['pred']
                    #         pred_pose = np.asarray(pred_pose)
                    #         pred_hrnet_pose.append(pred_pose.reshape(1,17,3))
                    #         NEW_PRED = NEW_PRED + 1
                    #         img_path_p = os.path.join('/Extra/panzhiyu/CMU_data',pred_2d_results[NEW_PRED]['image_name'])

                    if len(pred_bodys_2d) == 0:
                        continue

                    
                    pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scale, idx_v) # device pred_bodys_2d with relative depth
                    pred_bodys_3d, adjust_2d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scale, idx_v, Kd) # device
                    
                    # generate the pointcloud
                    pred_bodys_2d = adjust_2d.copy()
                    # test: replace the 2D pred with groundtruth
                    # refine_2dgt = refine_2d(adjust_2d, gt_bodys)


                    # replace the 2D rped with the HRNET
                    # pred_hrnet_pose = np.concatenate(pred_hrnet_pose, axis=0)
                    # pose_unified = pred_hrnet_pose[:,coco_to_unified,:] #panoptic_to_unified
                    # pose_neck = (pose_unified[:,1:2,:] + pose_unified[:,7:8,:])/2
                    # pose_14 = np.concatenate([pose_neck, pose_unified], axis=1)
                    # cmu_pred_2d = adjust_2d[:,[0]+panoptic_to_unified,:]
                    # adjust_2d = refine_2d_hrnet(cmu_pred_2d, pose_14)

                    # delete part of the low quality prediction
                    vis_label = (adjust_2d[:,:,3] > 1)
                    judge_num = np.sum(vis_label, axis=-1)
                    pred_valid = (judge_num >= 8)
                    adjust_2d = adjust_2d[pred_valid,...] # valid condition

                    # gt_label = (gt_bodys[:,:,3]>0)
                    # gt_num = np.sum(gt_label ,axis=-1)
                    # gt_valid = (gt_num >= 8)
                    # gt_bodys = gt_bodys[gt_valid,...]

                    # replace with the gt
                    # adjust_2d = refine_2dgt(adjust_2d, gt_bodys)
                    # if adjust_2d is None:
                    #     continue

                    # save the result
                    # gt_3d_ex = joints3d_global[i,...] # batch is 1
                    # nposes = torch.sum(gt_3d_ex[:,0,3]>0)
                    # gt_3d = gt_3d_ex[:nposes,...].cpu().numpy()

                    # pose2d_c.append(gt_bodys[:,:,[0,1,3]])
                    # cam_c.append(cam_p)

                    # save_result_cmu(pred_bodys_2d, adjust_2d, gt_bodys, gt_3d, scale['img_paths'][idx_v], cam_p, result) # judge the valid and add the feature
                    
                    # generate_pc(adjust_2d, cam_p, scale, batch_pc, i) # , device the total pointcloud of this frame in k lengt hpointcloud =
                    generate_pc_connection(adjust_2d, cam_p, scale, batch_pc, indx_match, select_2d_match, viewidx_match, view, i, shift_size, root_idx=0) # matching error? adjust_2d
                    pose_2d_collect[i][view] = torch.from_numpy(pred_bodys_2d[:,:,[0,1,3]]).to(device)
                

                # camera is same through different batch
                for k, v in cam_p.items():
                    cam_p[k] = torch.from_numpy(v).to(device).to(torch.float)
                
                cam_p['pro'] = cam_p['K'] @ torch.cat([cam_p['R'], cam_p['t']], dim=-1)
                cam_p['pro_inv'] = torch.pinverse(cam_p['pro'])
                
                cam_info[view] = cam_p #
                
                # smap_feature[view] = output_feature


            
            # end = time.time()
            # print(f'The every generation time is {end - start}s')

            # import pdb; pdb.set_trace()
                    # batch_pc[i].append(pointcloud)

                    # if refine_model is not None:
                    #     new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                    #                                                 device=device, root_n=cfg.DATASET.ROOT_IDX)
                    # else:

                    # new_pred_bodys_3d = pred_bodys_3d

                    # if cfg.TEST_MODE == "generate_train":
                    #     save_result_for_train_refine(pred_bodys_2d, new_pred_bodys_3d, gt_bodys, pred_rdepths, result)
                    # else:
                        # vis the data
                    # gt_3d amd new_pred_bodys_3d (分关节 和 总共)
                    
                    # val(torch.Tensor(pred_bodys_3d), torch.Tensor(gt_bodys[:,:,3:7]), mpjpe, kp_set)

                    # set the global coordinate to test the project

                    # if idx % 500 == 0:
                    #     gt_3d = gt_bodys[:,:,4:7]
                    #     joint_folder = os.path.join(output_dir,'joints_folder')
                    #     os.makedirs(joint_folder, exist_ok=True)
                    #     vis_joints3d(imgs[i], new_pred_bodys_3d, gt_3d, joint_folder, idx, i, idx_v, total_iter = total_iter)

                    # save_result(pred_bodys_2d, new_pred_bodys_3d, gt_bodys, pred_rdepths, result) #img_path[i],
                # if idx % 500 == 0:
                #     vis_2d(hmsIn, imgs[i], root_d_upsamp,output_dir,idx, idx_v, total_iter = total_iter)
            
            # get the batch pointcloud
            # set 1024 as the max per frame point num

            # do not considering the time cost
            # batch alignment for 1024

            # judge if it is the target 

            # generate the PC:
            failure_folder = os.path.join(output_dir,'vis_pose3d')
            rgb_folder = os.path.join(output_dir,'vis_rgb')
            debug_flag = 0
            os.makedirs(failure_folder, exist_ok=True)
            os.makedirs(rgb_folder, exist_ok=True)
            # pred_collect = []
            # mask_collect = []
            for k in range(cfg.DATASET.KEYPOINT.NUM): # TODO: can simplified
                for b in range(batch_size):
                    if len(batch_pc[k][b]) >= 1 : 
                        # batch_pc[k][b] = torch.cat(batch_pc[k][b], dim = 0)
                        batch_pc[k][b] = np.concatenate(batch_pc[k][b], axis = 0)
                        batch_pc[k][b] = torch.from_numpy(batch_pc[k][b]).to(device).unsqueeze(0)
                        indx_match[k][b] = np.concatenate(indx_match[k][b], axis = -1)
                        indx_match[k][b] = torch.from_numpy(indx_match[k][b]).to(device) # to_device
                        select_2d_match[k][b] = np.concatenate(select_2d_match[k][b], axis=0)
                        select_2d_match[k][b] = torch.from_numpy(select_2d_match[k][b]).to(device).unsqueeze(0)
                        viewidx_match[k][b] = np.concatenate(viewidx_match[k][b], axis = -1)
                        viewidx_match[k][b] = torch.from_numpy(viewidx_match[k][b]).to(device)
                       
                        # current_num = len(batch_pc[k][b])
                        # if current_num >= kp_align_num: # down_sampling
                        #     samp_idx = torch.randperm(current_num)[:kp_align_num]
                        #     batch_pc[k][b] = batch_pc[k][b][samp_idx,:]
                        #     batch_pc[k][b] = batch_pc[k][b].unsqueeze(0)
                        # else:
                        #     add_idx = torch.randperm(current_num)[:(kp_align_num - current_num)]
                        #     add_tensor = batch_pc[k][b][add_idx,:].clone()
                        #     batch_pc[k][b] = torch.cat([batch_pc[k][b],add_tensor], dim=0) # memory ?
                        #     batch_pc[k][b] = batch_pc[k][b].unsqueeze(0)

                        # bak_pred = pc_fusion(batch_pc[k][b])
                        # nposes = scales[i]['nposes']
                        # val_debug(bak_pred[0,...], joints3d_global[b,:nposes,k,:3], mpjpe_g3d, kp_set_g3d[k])
                    else:
                        # del batch_pc[k][b] # delete cannot align the result
                        debug_flag = 1
                        logger.info('Existing one batch contains no valid points')
                        # print(mpjpe_g3d.val)

                # if len(batch_pc[k]) >= 1:
                #     batch_pc[k] = torch.cat(batch_pc[k], dim=0) # TODO: need the further alignment
                # indx_match[k] = torch.cat(indx_match[k], dim=0)
                # select_2d_match[k] = torch.cat(select_2d_match[k],dim=0)
                # viewidx_match[k] = torch.cat(viewidx_match[k],dim=0)
                # else:
                #     debug_flag = 1
                # if len(batch_pc[k]) < 1:
                #     debug_flag = 1
            # 3D part and get the pose
            # if debug_flag:
            #     debug_flag = 0
            #     continue
            # it cannot do with the batch size because of the mis_alignment
            if debug_flag  == 1:
                debug_flag = 0
                continue

            pose_2d_related = dict() # all possible views
            for node in cfg.DATASET.CAM:
                view = list(node)[1]
                pose_2d_related[view] = torch.zeros(batch_size, max_people, cfg.DATASET.KEYPOINT.NUM, 7) #

            pred_bodys_3d = pc_fused_connection(batch_pc, indx_match, select_2d_match, viewidx_match, shift_size,cam_info, pose_2d_collect, pose_2d_related, device, root_idx=0)
            # current pred_3d is Batch * max_people * kp * 4
            # pred_num = pred_bodys_3d.shape[0]
            pred_bodys_3d = pred_bodys_3d.to(device)
            # find the corresponding matched_gt
            matched_pred, matched_gt, matched_pred_single = matchgt(pred_bodys_3d, joints3d_global, pose_2d_related)
            gt_bodys_2d = project_views(matched_gt, cam_info) # combine with the out of view information
            hm_collect = torch.cat(hm_collect, dim=0)
            # import pdb;pdb.set_trace()
            sampled_heatmap = project_views_samples(pred_bodys_3d, cam_info, scale, hm_collect)

            # for b in range(batch_size):
            #     matched_pred_b = matched_pred[b,...]
            #     pred_num = torch.sum(matched_pred_b[:,0,3]>0)
            #     matched_pred_b = matched_pred_b[:pred_num,]
            #     matched_gt_b = matched_gt[b,...]
            #     gt_num  = torch.sum(matched_gt_b[:,0,3]>0)
            #     matched_gt_b = matched_gt_b[:gt_num,...]
            #     val_match(matched_pred_b,matched_gt_b,mpjpe_g3d)
            # if mpjpe_g3d.avg > 100:
            #     mpjpe_g3d.reset()
            # import pdb;pdb.set_trace()
            # print('aaa')

            # save matched_single, pred3d, gt 2d and 3d, and one cam_info, divide by batch_size
            gnn_pair = dict()  # batch 4 pair
            gnn_pair['pred_single'] = matched_pred_single
            gnn_pair['pred_3d'] = matched_pred.cpu()
            gnn_pair['gt_3d'] = matched_gt.cpu()
            gnn_pair['gt_2d'] = gt_bodys_2d
            gnn_pair['cam'] = cam_info
            gnn_pair['hmIn'] = sampled_heatmap
            for k in gnn_pair['gt_2d'].keys():
                gnn_pair['gt_2d'][k] = gnn_pair['gt_2d'][k].cpu()
                for n_k in gnn_pair['cam'][k].keys():
                    gnn_pair['cam'][k][n_k] = gnn_pair['cam'][k][n_k].cpu()

            file_name = f'gnn_train_{save_idx:0>5d}.pkl'
            with open(os.path.join(save_dir,file_name), 'wb') as f:
                pickle.dump(gnn_pair, f)
            save_idx += 1
            

            # # the refinenet
            # _, refine_res = gnn_net(pred_bodys_2d, smap_feature, cam_info)
            # refined_pose3d = torch.cat([matched_pred[:,:,:,:3] + refine_res, matched_pred[:,:,:,3:4]], dim = -1)
            # # caluculate the loss
            # pred_count = torch.sum( refined_pose3d[:,:,0,3],dim=-1, keepdim=True )
            # valid_label = refined_pose3d[:,:,:,3:4] * (matched_gt[:,:,:,3:4] > 0.1)
            # loss_temp = torch.norm((refined_pose3d[:,:,:,:3] - matched_gt[:,:,:,:3]) * valid_label, dim=-1) ** 2
            # loss_m1 = torch.mean(loss_temp, dim=-1)         
            # loss_3d = torch.mean(torch.sum(loss_m1 * (1/pred_count), dim = -1))
            
            
            
            
            # nposes = torch.sum(gt_3d_ex[:,0,3]>0)
            # gt_3d = gt_3d_ex[:nposes,...]
            # sample_out = dict()
            # sample_out['heatmap'] = hm_collect
            # sample_out['gt_2d'] = pose2d_c
            # sample_out['camera'] = cam_c
            # sample_out['pred_3d'] = pred_bodys_3d
            # sample_out['gt_3d'] = gt_3d
            # output_file = 'sample_output.pkl'
            # with open(output_file , 'wb') as f:
            #     pickle.dump(sample_out, f)
            # gt_3d = gt_3d[:,[0]+panoptic_to_unified,:]

            # vis_label = val_match(pred_bodys_3d,gt_3d,mpjpe) # validation code

            # gt_3d = gt_3d[:,[0]+panoptic_to_unified,:] # for hrnet
            # vis_label_fn = val_fn(pred_bodys_3d, gt_3d, mpjpe, kp_set) # FN
            # vis_label_fp = val_fp(pred_bodys_3d, gt_3d, mpjpe_g3d, kp_set_g3d) # FP

            # fig = plt.figure(figsize=(20, 10))
            # ax1 = fig.add_subplot(111, projection='3d')
            # for ip in range(len(pred_bodys_3d)):
            #     p3d = pred_bodys_3d[ip]
            #     for pair in pairs:
            #         ax1.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)], linestyle= ':') # linestyle=':'
            # for ip in range(len(gt_3d)):
            #     p3d = gt_3d[ip]
            #     for pair in pairs:
            #         ax1.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
            # ax1.view_init(azim=-90, elev=-45)
            # # file_name = os.path.join(failure_folder, f'vis_f_fp_{idx}.jpg')
            # plt.savefig('gt2d_debug.jpg')
            # import pdb;pdb.set_trace()
            # if vis_label:
            #     # plot the rgb
            #     fig = plt.figure(figsize=(20, 10))
            #     img_c = []
            #     for idx_v ,node in enumerate(cfg.DATASET.CAM):
            #         img = imgs_mul[0,idx_v,...].cpu()
            #         img_c.append(tensor2im(img))
            #     img_c = np.concatenate(img_c, axis=1)
            #     plt.imshow(img_c)
            #     rgb_file = os.path.join(rgb_folder, f'vis_f_{idx}.jpg')
            #     plt.savefig(rgb_file)
            #     fig = plt.figure(figsize=(20, 10))
            #     ax1 = fig.add_subplot(111, projection='3d')
            #     for ip in range(len(pred_bodys_3d)):
            #         p3d = pred_bodys_3d[ip]
            #         for pair in pairs:#unified_bones_def14
            #             ax1.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)], linestyle= ':') # linestyle=':'
            #     for ip in range(len(gt_3d)):
            #         p3d = gt_3d[ip]
            #         for pair in pairs:
            #             ax1.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
            #     ax1.view_init(azim=-90, elev=-45)
            #     file_name = os.path.join(failure_folder, f'vis_f_{idx}.jpg')
            #     plt.savefig(file_name)
            #     plt.close(fig)
            # except:
            #     debug_flag = 0
            #     print('The debug flag is set True')

            # pred_centers, nms_mask, group_inds = pc_fused(batch_pc[0].float())
            #  index according to the fusing data 

            # # create the whole pose
            # if len(batch_pc[k]) >= 1: # TODO : for analysis  maybe course the mismatch of the batch dimension
            #     batch_pc[k] = torch.cat(batch_pc[k], dim=0) # TODO: current batch size = 1
            #     # cuda the part
            #     pred_centers, nms_mask = pc_fused(batch_pc[k].float())

            #     val_flag, npose = val_debug(pred_centers, nms_mask, joints3d_global, k, mpjpe_g3d, kp_set_g3d[k], fp_total, kp_fp[k]) # test the  AP AR MPJPE 

            #     pred_collect.append(pred_centers[None,...])
            #     mask_collect.append(nms_mask[None,...])
            #     # # for center, mask in zip(bak_center, nms_mask):
            #     # val_flag, npose = val_debug(pred_centers, nms_mask, joints3d_global, k, mpjpe_g3d, kp_set_g3d[k], fp_total, kp_fp[k]) # test the  AP AR MPJPE 
            #     # # TODO: set batch = 1 to analysis the falure case
                
            #     # if val_flag:
            #     #     kp_extracted = joints3d_global[0,:npose,k,:3]
            #     #     fig = plt.figure(figsize=(20, 15))
            #     #     # ax = fig.add_subplot(121,projection='3d') #
            #     #     ax = fig.add_subplot(111, projection='3d')
            #     #     ax.scatter(batch_pc[k][0,:,0].cpu().numpy(), batch_pc[k][0,:,1].cpu().numpy(), batch_pc[k][0,:,2].cpu().numpy(), marker='o', s = 4)
            #     #     ax.scatter(pred_centers[0,:,0].cpu().numpy(), pred_centers[0,:,1].cpu().numpy(), pred_centers[0,:,2].cpu().numpy(), marker='^',s = 320,c='k')
            #     #     ax.scatter(kp_extracted[:,0].cpu().numpy(), kp_extracted[:,1].cpu().numpy(), kp_extracted[:,2].cpu().numpy(),marker='o',s=880)
            #     #     file_name = os.path.join(failure_folder, f'error_j_{key_name[k]}_f_{idx}.jpg')
            #     #     plt.savefig(file_name)
            #     #     plt.close()

            # TODO : debug
            # try:
            # pred_collect = torch.cat(pred_collect , dim=0)
            # mask_collect = torch.cat(mask_collect , dim=0)
            # pred_bodys_3d = PoseSolver(pred_collect, mask_collect)
            # for b in range(batch_size):
            #     pred_pose = pred_bodys_3d[b]
            #     gt_3d_ex = joints3d_global[b,...]
            #     nposes = torch.sum(gt_3d_ex[:,0,3]>0)
            #     gt_3d = gt_3d_ex[:nposes,...]
            #     val_v2(pred_pose, gt_3d, mpjpe, kp_set) 

                # fig = plt.figure(figsize=(20, 10))
                # ax1 = fig.add_subplot(111, projection='3d')
                # for ip in range(len(pred_pose)):
                #     p3d = pred_pose[ip]
                #     for pair in pairs:
                #         ax1.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)], linestyle= ':') # linestyle=':'
                # # for ip in range(len(gt_3d)):
                # #     p3d = gt_3d[ip]
                # #     for pair in pairs:
                # #         ax1.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
                # ax1.view_init(azim=-90, elev=-45)
                # file_name = os.path.join(failure_folder, f'vis_f_{idx}.jpg')
                # plt.savefig(file_name)
            
            # except:
            #     print('The mis alignment')


             
            
            

            # write the pointcloud for debug ,and joints3d_global
            # with open('pointcloud_kpdebug_tc.pkl', 'wb') as f:
            #     pickle.dump(batch_pc, f)
            # with open('jointgt_debug.pkl', 'wb') as f:
            #     pickle.dump(joints3d_global, f)
            
            # import pdb; pdb.set_trace()



    # save the final result in json file

    dir_name = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]
    pair_file_name = os.path.join(output_dir, '{}_{}_{}_{}orig.pkl'.format(dir_name, cfg.TEST_MODE,
                                                                        cfg.DATA_MODE, cfg.JSON_SUFFIX_NAME)) # consider the orig
    # with open(pair_file_name, 'w') as f:
    #     json.dump(result, f)
    with open(pair_file_name, 'wb') as f:
        pickle.dump(result, f)
    logger.info("Pairs writed to {}".format(pair_file_name))

    msg =  f'EPOCH {total_iter}: The MPJPE is {mpjpe.avg}'  #
    logger.info(msg)
    # for i in range(kpt_num):
    #     logger.info(f'The {key_name[i]} is {kp_set[i].avg}') # fn

    # msg =  f'EPOCH {total_iter}: The FP_MPJPE is {mpjpe_g3d.avg}'  #_g3d fp
    # logger.info(msg)
    # for i in range(kpt_num):
    #     logger.info(f'The {key_name[i]} is {kp_set_g3d[i].avg}') #_g3d

    return mpjpe.avg

def matchgt(pred_3d_batch, gt_3d_batch, pose2d_related):
    batch_size = pred_3d_batch.shape[0]
    matched_pred = torch.zeros(pred_3d_batch.shape).to(pred_3d_batch.device)
    matched_gt = torch.zeros(pred_3d_batch.shape).to(pred_3d_batch.device)
    matched_pred2d = dict()
    for v in pose2d_related.keys():
        matched_pred2d[v] = torch.zeros(pose2d_related[v].shape).to(pose2d_related[v].device)

    for b in range(batch_size):
        pred_3d_b = pred_3d_batch[b,...]
        gt_3d = gt_3d_batch[b,...]
        pred_3d = pred_3d_b[...,:3] # N X K X 3
        pred_vis = pred_3d_b[...,3]
        gt_3d_position = gt_3d[...,:3]
        gt_vis = gt_3d[...,3]
        mpjpe_save = dict()
        mpjpe_id = dict()
        for idx, (pose, pose_vis) in enumerate(zip(pred_3d, pred_vis)):
            temp_mpjpe = []
            for (gt, gt_valid) in zip(gt_3d_position, gt_vis):
                mask1 = (gt_valid > 0)
                mask2 = (pose_vis > 0)
                mask = mask1 * mask2
                if torch.sum(mask) == 0:
                    temp_mpjpe.append(torch.tensor(550))
                    continue
                c_mpjpe = torch.mean(torch.norm((pose[mask,:] - gt[mask,:]), dim=-1))
                temp_mpjpe.append(c_mpjpe)
            min_gt = torch.argmin(torch.Tensor(temp_mpjpe))
            min_mpjpe = torch.min(torch.Tensor(temp_mpjpe))
            if min_gt.item() in mpjpe_save.keys():
                if min_mpjpe < mpjpe_save[min_gt.item()]:
                    mpjpe_save[min_gt.item()] = min_mpjpe
                    mpjpe_id[min_gt.item()] = idx
            else:
                mpjpe_save[min_gt.item()] = min_mpjpe
                mpjpe_id[min_gt.item()] = idx
        # error_list = torch.Tensor(list(mpjpe_save.values()))
        # mask_label = (error_list < threshold)
        filtered_predid = list(mpjpe_id.values())
        filtered_pose = pred_3d_b[filtered_predid,...]
        filtered_gtid = list(mpjpe_id.keys())
        filtered_gt = gt_3d[filtered_gtid,...]
        pred_num = len(filtered_predid)
        matched_pred[b,:pred_num,...] = filtered_pose
        matched_gt[b,:pred_num,...] = filtered_gt
        # match the 2d
        for v, v_pose in pose2d_related.items():
            filtered_pose2d = v_pose[b,filtered_predid,...]
            matched_pred2d[v][b,:pred_num,...] = filtered_pose2d
    
    return matched_pred, matched_gt, matched_pred2d




        


def easy_mode(pose_file, poserefine, cfg, logger, device, 
                            output_dir='', total_iter='infer'):
    if poserefine is not None:
        poserefine.eval()
    os.makedirs(output_dir, exist_ok=True)
    # pred_2d_results
    with open(pose_file,'rb') as f:
        ready_pose = pickle.load(f)
        ready_pose = ready_pose['3d_pairs']
    total_len = len(ready_pose)
    root_path = 'Extra/panzhiyu/CMU_data/'
    IDX_sort = 0   #7812 especial
    shift_size = 3 # shift size is 1
    scale = dict()
    scale['img_width'] = 1920
    scale['img_height'] = 1080
    mpjpe = AverageMeter()
    mpjpe_refined = AverageMeter()
    mpjpe_refined2 = AverageMeter()
    precision = FPAverageMeter()
    recall = FPAverageMeter()
    precision_r = FPAverageMeter()
    recall_r = FPAverageMeter()
    precision_r2 = FPAverageMeter()
    recall_r2 = FPAverageMeter()
    idx = 1
    # IDX_sort = 3576
    while IDX_sort < total_len:
        if idx % 100 == 0:
            logger.info(f'Process {idx}, IDX is {IDX_sort}/{total_len}: The current MPJPE is {mpjpe.avg}, the opt is {mpjpe_refined.avg}')
        # get one scene 
        batch_pc = [[[] for _ in range(1)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        indx_match = [[[] for _ in range(1)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        select_2d_match = [[[] for _ in range(1)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        # use the view match, tag the view label
        viewidx_match = [[[] for _ in range(1)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]

        per_frame_info = dict()
        init_image = ready_pose[IDX_sort]['image_path']
        gt_joints = np.array(ready_pose[IDX_sort]['gt_3d'])
        view_r = init_image.split('/')[-2]
        image_num = init_image.split('/')[-1].replace(view_r+'_','')
        per_frame_info[view_r] = ready_pose[IDX_sort]
        IDX_sort = IDX_sort + 1
        if IDX_sort >= total_len:
            msg =  f'EPOCH {total_iter}: The MPJPE is {mpjpe.avg}'  #
            logger.info(msg)  
            msg =  f'EPOCH {total_iter}: The MPJPE_opt is {mpjpe_refined.avg}'  #
            logger.info(msg)
            break
        next_image = ready_pose[IDX_sort]['image_path']
        next_viewr = next_image.split('/')[-2]
        next_num = next_image.split('/')[-1].replace(next_viewr+'_','')
        while image_num == next_num:
            per_frame_info[next_viewr] = ready_pose[IDX_sort]
            image_num = next_num
            IDX_sort = IDX_sort + 1
            if IDX_sort >= total_len:
                msg =  f'EPOCH {total_iter}: The MPJPE is {mpjpe.avg}'  #
                logger.info(msg)  
                msg =  f'EPOCH {total_iter}: The MPJPE_opt is {mpjpe_refined.avg}'  #
                logger.info(msg)
                break 
            next_image = ready_pose[IDX_sort]['image_path']
            next_viewr = next_image.split('/')[-2]
            next_num = next_image.split('/')[-1].replace(next_viewr+'_','')
        
        # process the information
        # pose_2d_collect = dict()
        pose_2d_collect = [dict() for _ in range(1)]
        vis_gt_2d_collect = []
        vis_pred_collect = []
        vis_img_path = []
        cam_info = dict()
        # according to this unit
        for view_idx ,(view, info) in  enumerate(per_frame_info.items()):
            # get the cam_paras
            cam_p = info['cam']
            pose_2d = info['pred_aligned']  # It is reduced
            gt_bodys = info['gt_2d']
            image_path = info['image_path']
            # match the hrnet

            # idx_path = image_path.split(root_path)[-1]
            # if idx_path in pred_2d_results.keys():
            #     hrnet_pose = pred_2d_results[idx_path]
            #     pose_unified = hrnet_pose[:,coco_to_unified,:]
            #     neck_vis = (pose_unified[:,1:2,2]) * (pose_unified[:,7:8,2])
            #     pose_neck = (pose_unified[:,1:2,:] + pose_unified[:,7:8,:])/2
            #     pose_neck[:,:,2] = neck_vis
            #     pose_14 = np.concatenate([pose_neck, pose_unified], axis=1)
            #     cmu_pred_2d = pose_2d[:,[0]+panoptic_to_unified,:]
            #     # process the vis label of hrnet
            #     adjust_2d = refine_2d_hrnet(cmu_pred_2d, pose_14)
            #     if lenadjust_2d is None:
            #         continue
            #     pose_2d = adjust_2d.copy() # replace the original prediction
            # else:
            #     continue
            
            # match the gt pred

            # adjust_2d = refine_2dgt(pose_2d, gt_bodys)
            # if len(adjust_2d) == 0  or adjust_2d is None:
            #     continue
            # pose_2d = adjust_2d.copy()

            view = int(view.split('_')[-1])
            pose_2d_collect[0][view] =torch.from_numpy(pose_2d[:,:,[0,1,3]]).to(device) # (x, y, vis)

            # cam_info.append(cam_p) # for alignment
            vis_gt_2d_collect.append(gt_bodys[:,:,[0,1,3]])
            vis_pred_collect.append(pose_2d[:,:,[0,1,3]])
            vis_img_path.append(image_path)
            generate_pc_connection(pose_2d, cam_p, scale, batch_pc, indx_match, select_2d_match, viewidx_match, view, 0, shift_size, root_idx=0) #adjust_2dcmu_pred_2d
            
            for k, v in cam_p.items():
                cam_p[k] = torch.from_numpy(v).to(device).to(torch.float)
            cam_info[view] = cam_p

        
        failure_3d_folder = os.path.join(output_dir,'vis_pose3d_1009D')
        failure_2d_folder = os.path.join(output_dir,'vis_pose2d_1009D')
        debug_flag = 0
        os.makedirs(failure_3d_folder, exist_ok=True)
        os.makedirs(failure_2d_folder, exist_ok=True)
        # pred_collect = []
        # mask_collect = []
        for k in range(cfg.DATASET.KEYPOINT.NUM): # TODO: #cfg.DATASET.KEYPOINT.NUM
            for b in range(1):
                if len(batch_pc[k][b]) >= 1 : 
                    # batch_pc[k][b] = torch.cat(batch_pc[k][b], dim = 0)
                    batch_pc[k][b] = np.concatenate(batch_pc[k][b], axis = 0)
                    batch_pc[k][b] = torch.from_numpy(batch_pc[k][b]).to(device).unsqueeze(0)
                    select_2d_match[k][b] = np.concatenate(select_2d_match[k][b], axis=0)
                    select_2d_match[k][b] = torch.from_numpy(select_2d_match[k][b]).to(device).unsqueeze(0)
                    indx_match[k][b] = np.concatenate(indx_match[k][b], axis = -1)
                    indx_match[k][b] = torch.from_numpy(indx_match[k][b]).to(device) # to_device    
                    viewidx_match[k][b] = np.concatenate(viewidx_match[k][b], axis = -1)
                    viewidx_match[k][b] = torch.from_numpy(viewidx_match[k][b]).to(device)
            #     else:
            #         del batch_pc[k][b]
            #         # print(mpjpe_g3d.val)
            # if len(batch_pc[k]) >= 1: # no points in one 
            #     batch_pc[k] = torch.cat(batch_pc[k], dim=0) # TODO: need the further alignment
            #     select_2d_match[k] = torch.cat(select_2d_match[k],dim=0)
            #     indx_match[k] = torch.cat(indx_match[k], dim=0)
            #     viewidx_match[k] = torch.cat(viewidx_match[k],dim=0)
            # else:
            #     debug_flag = 1 # TODO: 可以稍微放松要求
            if len(batch_pc[k]) < 1:
                debug_flag = 1
        
        # 3D part and get the pose
        # try:
        if debug_flag:
            # assert False
            debug_flag = 0
            continue
        
        pose_2d_related = dict() # all possible views
        for node in cfg.DATASET.CAM:
            view = list(node)[1]
            pose_2d_related[view] = torch.zeros(1, 10, cfg.DATASET.KEYPOINT.NUM, 7) #
        
        pred_bodys_3d = pc_fused_connection(batch_pc, indx_match, select_2d_match, viewidx_match, shift_size, cam_info, pose_2d_collect, pose_2d_related, device, root_idx=0) # changed the 
        for node in cfg.DATASET.CAM:
            view = list(node)[1]
            pose_2d_related[view] = pose_2d_related[view].to(device)
        pred_bodys_3d = pred_bodys_3d.to(device)
        with torch.no_grad():
            if poserefine is not None:
                refine_bodys_3d, _, _ = poserefine(pose_2d_related, pred_bodys_3d) # It has the single pred PAY ATENTION TO THIS
        refine_bodys_3d = torch.cat([refine_bodys_3d, pred_bodys_3d[...,3:4]],dim=-1)
        with torch.no_grad():
            if poserefine is not None:
                refine_bodys_3d2, _, _ = poserefine(pose_2d_related, refine_bodys_3d) # It has the single pred PAY ATENTION TO THIS
        refine_bodys_3d2 = torch.cat([refine_bodys_3d2, pred_bodys_3d[...,3:4]],dim=-1)
        pred_bodys_3d = pred_bodys_3d[0,...]
        refine_bodys_3d = refine_bodys_3d[0,...]
        refine_bodys_3d2 = refine_bodys_3d2[0,...]
        pred_num = torch.sum(pred_bodys_3d[:,0,3]>0)
        if pred_num  == 0:
            continue
        pred_bodys_3d = pred_bodys_3d[:pred_num,...]
        refine_bodys_3d = refine_bodys_3d[:pred_num,...]
        refine_bodys_3d2 = refine_bodys_3d2[:pred_num,...]
        gt_3d = torch.from_numpy(gt_joints)  #joints3d_global[0,...] # batch is 1
        # gt_3d = gt_3d[:,[0]+panoptic_to_unified,:]  # It is for hrnet standard
        vis_label, filtered_pose = val_match(pred_bodys_3d.cpu(),gt_3d,mpjpe, precision, recall, ap_threshold=2.5) # output
        _,_ = val_match(refine_bodys_3d.cpu(),gt_3d,mpjpe_refined, precision_r, recall_r, ap_threshold=2.5)
        _,_ = val_match(refine_bodys_3d2.cpu(),gt_3d,mpjpe_refined2,precision_r2, recall_r2, ap_threshold=2.5) # useless 
        if np.isnan(np.array(mpjpe.val)):
            import pdb; pdb.set_trace()


        # # refined
        # filtered_pose = filtered_pose.to(device)
        # pose_2d_collect = pose_2d_collect[0]
        # # pose_2d_collect = list(pose_2d_collect.values())
        # # cam_info = list(cam_info.values())
        # refined_3d = optimize_3d(filtered_pose, pose_2d_collect, cam_info, device)    
        # vis_label, _ = val_match(refined_3d.cpu(),gt_3d,refine_bodys_3d)
        # if np.isnan(np.array(mpjpe_refined.val)):
        #     import pdb; pdb.set_trace()
        # refined_3d = refined_3d.detach().cpu().numpy()

        # if vis_label:
        #     msg =  f'BUG EPOCH {total_iter}: The MPJPE is {mpjpe.val}'  #
        #     logger.info(msg)  
        #     msg =  f'BUG EPOCH {total_iter}: The MPJPE_opt is {mpjpe_refined.val}'  #
        #     logger.info(msg)  
        #     # vis_2d_pose
        #     fig1 = plt.figure(figsize=(20, 10))
        #     fig2 = plt.figure(figsize=(20, 10))
        #     views_num = len(vis_pred_collect)
        #     for i in range(views_num):
        #         ax1 = fig1.add_subplot(3,2,(i+1))
        #         pose_2d = vis_pred_collect[i]
        #         gt_2d = vis_gt_2d_collect[i]
        #         if len(pose_2d) == 0:
        #             continue
        #         img = cv2.imread(vis_img_path[i])
        #         img = img[...,::-1]
        #         ax1.imshow(img)
        #         for ip in range(len(pose_2d)):
        #             p3d = pose_2d[ip]
        #             for pair in unified_bones_def14: #unified_bones_def14pairs
        #                 if np.all(p3d[pair,2] > 0):
        #                     ax1.plot(p3d[pair, 0], p3d[pair, 1], c=colors[ip%len(colors)])

        #         ax2 = fig2.add_subplot(3,2,(i+1))
        #         ax2.imshow(img)
        #         for ip in range(len(gt_2d)):
        #             p3d = gt_2d[ip]
        #             for pair in unified_bones_def14: #unified_bones_def14
        #                 if np.all(p3d[pair,2] > 0):
        #                     ax2.plot(p3d[pair, 0], p3d[pair, 1], c=colors[ip%len(colors)])
                
        #     # orig pose visualization
        #     # fig3 = plt.figure(figsize=(20, 10))
        #     # ax3 = fig3.add_subplot(111, projection='3d')
        #     # for ip in range(len(refined_3d)): #pred_bodys_3d
        #     #     p3d = refined_3d[ip]
        #     #     for pair in unified_bones_def14: #unified_bones_def14
        #     #         ax3.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)], linestyle= ':') # linestyle=':'
        #     # for ip in range(len(gt_3d)):
        #     #     p3d = gt_3d[ip]
        #     #     for pair in unified_bones_def14:
        #     #         ax3.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
        #     # ax3.view_init(azim=-90, elev=-45)

        #     fig4 = plt.figure(figsize=(20, 10))
        #     ax4 = fig4.add_subplot(111, projection='3d')
        #     for ip in range(len(pred_bodys_3d)): #pred_bodys_3d
        #         p3d = pred_bodys_3d[ip]
        #         for pair in unified_bones_def14: #unified_bones_def14
        #             ax4.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)], linestyle= ':') # linestyle=':'
        #     for ip in range(len(gt_3d)):
        #         p3d = gt_3d[ip]
        #         for pair in unified_bones_def14:
        #             ax4.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
        #     ax4.view_init(azim=-90, elev=-45)

        #     file_name = os.path.join(failure_2d_folder, f'vis_p2d_{idx}.jpg')
        #     fig1.savefig(file_name)
        #     plt.close(fig1)
        #     file_name = os.path.join(failure_2d_folder, f'vis_gt2d_{idx}.jpg')
        #     fig2.savefig(file_name)
        #     plt.close(fig2)
        #     # file_name = os.path.join(failure_3d_folder, f'vis_3dopt_{idx}.jpg')
        #     # fig3.savefig(file_name)
        #     # plt.close(fig3)
        #     file_name = os.path.join(failure_3d_folder, f'vis_3dorg_{idx}.jpg')
        #     fig4.savefig(file_name)
        #     plt.close(fig4)

        idx = idx + 1
        # except:
        #     if debug_flag == 0:
        #         print(f'Else error, the IDX_sort is {IDX_sort}, idx is {idx}')
        #     print(f'Flag is {debug_flag}, The debug flag is set True')
        #     debug_flag = 0
            
    msg =  f'EPOCH {total_iter}: The MPJPE is {mpjpe.avg}'  #
    logger.info(msg)  
    msg =  f'EPOCH {total_iter}: The MPJPE_opt is {mpjpe_refined.avg}'  #
    logger.info(msg) 
    return  mpjpe_refined.avg

def easy_mode_cross(pose_file, poserefine, cfg, logger, device, 
                            output_dir='', total_iter='infer'):
    if poserefine is not None:
        poserefine.eval()
    os.makedirs(output_dir, exist_ok=True)
    # pred_2d_results
    with open(pose_file,'rb') as f:
        ready_pose = pickle.load(f)

    view_num = 3
    cam_file = '/Extra/panzhiyu/CampusSeq1/cam_para.pkl'  # for campus test
    with open(cam_file, 'rb') as f:
        cam_para = pickle.load(f)
    init_num = 2000
    max_length = 0
    for v in range(view_num):
        c_frame = list(ready_pose[v].keys())[0]
        c_len = len(ready_pose[v].keys())
        if c_len > max_length:
            max_length = c_len
        if init_num > c_frame:
            init_num = c_frame

    shift_size = 3 # shift size is 1
    scale = dict()
    scale['img_width'] = 360 # TODO 1032
    scale['img_height'] = 288 # 776
    mpjpe = AverageMeter()
    mpjpe_orig = [AverageMeter() for _ in range(view_num)]
    precision = FPAverageMeter()
    recall = FPAverageMeter()
    precision_orig = [FPAverageMeter() for _ in range(view_num)]
    recall_orig = [FPAverageMeter() for _ in range(view_num)]

    for i in tqdm(range(max_length)):
        current_num = init_num + i
        per_frame_info = dict()
        for v in range(view_num):
            if current_num in ready_pose[v].keys():
                per_frame_info[v] = ready_pose[v][current_num]
                gt_joints = np.array(ready_pose[v][current_num]['gt_global'])
        c_view_num = len(per_frame_info.keys())
        if c_view_num <= 1: 
            continue
        pose_2d_collect = [dict() for _ in range(1)]
        vis_gt_2d_collect = []
        vis_pred_collect = []
        vis_img_path = []
        cam_info = dict()
        batch_pc = [[[] for _ in range(1)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        indx_match = [[[] for _ in range(1)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        select_2d_match = [[[] for _ in range(1)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        # use the view match, tag the view label
        viewidx_match = [[[] for _ in range(1)] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        for view_idx ,(view, info) in  enumerate(per_frame_info.items()):
            # get the cam_paras
            # cam_p = info['cam']
            cam_p = cam_para[view].copy() 
            pose_2d = info['pred']  # It is reduced
            gt_bodys = info['gt_local']
            image_path = info['img_path']
            
            # match the hrnet

            # idx_path = image_path.split(root_path)[-1]
            # if idx_path in pred_2d_results.keys():
            #     hrnet_pose = pred_2d_results[idx_path]
            #     pose_unified = hrnet_pose[:,coco_to_unified,:]
            #     neck_vis = (pose_unified[:,1:2,2]) * (pose_unified[:,7:8,2])
            #     pose_neck = (pose_unified[:,1:2,:] + pose_unified[:,7:8,:])/2
            #     pose_neck[:,:,2] = neck_vis
            #     pose_14 = np.concatenate([pose_neck, pose_unified], axis=1)
            #     cmu_pred_2d = pose_2d[:,[0]+panoptic_to_unified,:]
            #     # process the vis label of hrnet
            #     adjust_2d = refine_2d_hrnet(cmu_pred_2d, pose_14)
            #     if lenadjust_2d is None:
            #         continue
            #     pose_2d = adjust_2d.copy() # replace the original prediction
            # else:
            #     continue
            
            # match the gt pred

            # adjust_2d = refine_2dgt(pose_2d, gt_bodys)
            # if len(adjust_2d) == 0  or adjust_2d is None:
            #     continue
            # pose_2d = adjust_2d.copy()

            # view = int(view.split('_')[-1]) # view is already be the number
            pose_2d_collect[0][view] =torch.from_numpy(pose_2d[:,:,[0,1,3]]).to(device) # (x, y, vis)
            # cam_info.append(cam_p) # for alignment
            vis_gt_2d_collect.append(gt_bodys[:,:,[0,1,3]])
            vis_pred_collect.append(pose_2d[:,:,[0,1,3]])
            vis_img_path.append(image_path)
            temp_2d = pose_2d.reshape(-1,4)
            temp_3d = back_to_global(temp_2d[:,:2],temp_2d[:,2],cam_p['K'], cam_p['distCoef'],cam_p['R'], cam_p['t'])
            single_3d = temp_3d.reshape(-1,15,3) * 100
            single_3d = np.concatenate([single_3d, pose_2d[...,3:4]], axis=-1)
            single_3d = torch.from_numpy(single_3d)
            # 3D COMPARE
            gt_3d = torch.from_numpy(gt_joints)  #joints3d_global[0,...] # batch is 1
            judge = (torch.sum((gt_3d != 0), dim=-1) > 0 )
            gt_num = torch.sum(torch.any(judge, dim=-1))
            gt_3d = torch.cat([gt_3d, judge.unsqueeze(-1).float()], dim=-1)
            gt_3d = gt_3d[:gt_num, ...]
            gt_3d[...,:3] = gt_3d[...,:3] * 100
            _,_ = val_match(single_3d,gt_3d,mpjpe_orig[view], precision_orig[view], recall_orig[view])


            generate_pc_connection(pose_2d, cam_p, scale, batch_pc, indx_match, select_2d_match, viewidx_match, view, 0, shift_size, root_idx=2) #adjust_2dcmu_pred_2d

            for k, v in cam_p.items():
                cam_p[k] = torch.from_numpy(v).to(device).to(torch.float)
            cam_info[view] = cam_p

        
        failure_3d_folder = os.path.join(output_dir,'vis_pose3d_1009D')
        failure_2d_folder = os.path.join(output_dir,'vis_pose2d_1009D')
        debug_flag = 0
        os.makedirs(failure_3d_folder, exist_ok=True)
        os.makedirs(failure_2d_folder, exist_ok=True)
        # pred_collect = []
        # mask_collect = []
        for k in range(cfg.DATASET.KEYPOINT.NUM): # TODO: #cfg.DATASET.KEYPOINT.NUM
            for b in range(1):
                if len(batch_pc[k][b]) >= 1 : 
                    # batch_pc[k][b] = torch.cat(batch_pc[k][b], dim = 0)
                    batch_pc[k][b] = np.concatenate(batch_pc[k][b], axis = 0)
                    batch_pc[k][b][:,:3] = batch_pc[k][b][:,:3] * 100
                    batch_pc[k][b] = torch.from_numpy(batch_pc[k][b]).to(device).unsqueeze(0)
                    select_2d_match[k][b] = np.concatenate(select_2d_match[k][b], axis=0)
                    select_2d_match[k][b][:,:3] = select_2d_match[k][b][:,:3] * 100
                    select_2d_match[k][b] = torch.from_numpy(select_2d_match[k][b]).to(device).unsqueeze(0)
                    indx_match[k][b] = np.concatenate(indx_match[k][b], axis = -1)
                    indx_match[k][b] = torch.from_numpy(indx_match[k][b]).to(device) # to_device    
                    viewidx_match[k][b] = np.concatenate(viewidx_match[k][b], axis = -1)
                    viewidx_match[k][b] = torch.from_numpy(viewidx_match[k][b]).to(device)
            #     else:
            #         del batch_pc[k][b]
            #         # print(mpjpe_g3d.val)
            # if len(batch_pc[k]) >= 1: # no points in one 
            #     batch_pc[k] = torch.cat(batch_pc[k], dim=0) # TODO: need the further alignment
            #     select_2d_match[k] = torch.cat(select_2d_match[k],dim=0)
            #     indx_match[k] = torch.cat(indx_match[k], dim=0)
            #     viewidx_match[k] = torch.cat(viewidx_match[k],dim=0)
            # else:
            #     debug_flag = 1 # TODO: 可以稍微放松要求

            if len(batch_pc[k]) < 1:
                debug_flag = 1
        
        # 3D part and get the pose
        # try:
        if debug_flag:
            # assert False
            debug_flag = 0
            continue

        pose_2d_related = dict() # all possible views
        for node in range(view_num):
            pose_2d_related[node] = torch.zeros(1, 10, cfg.DATASET.KEYPOINT.NUM, 7) #
        
        pred_bodys_3d = pc_fused_connection(batch_pc, indx_match, select_2d_match, viewidx_match, shift_size, cam_info, pose_2d_collect, pose_2d_related, device, root_idx=2) # changed the 
        pred_bodys_3d = pred_bodys_3d.to(device)
        pred_bodys_3d = pred_bodys_3d[0,...]
        pred_num = torch.sum(pred_bodys_3d[:,0,3]>0)
        if pred_num  == 0:
            continue
        pred_bodys_3d = pred_bodys_3d[:pred_num,...]
        gt_3d = torch.from_numpy(gt_joints)  #joints3d_global[0,...] # batch is 1
        judge = (torch.sum((gt_3d != 0), dim=-1) > 0 )
        gt_num = torch.sum(torch.any(judge, dim=-1))
        gt_3d = torch.cat([gt_3d, judge.unsqueeze(-1).float()], dim=-1)
        gt_3d = gt_3d[:gt_num, ...]
        gt_3d[...,:3] = gt_3d[...,:3] * 100
        vis_label, filtered_pose = val_match(pred_bodys_3d.cpu(),gt_3d,mpjpe, precision, recall) # output # unit is meter  
        # for view in per_frame_info.keys():
        #     cmp_3d = pose_2d_related[view][0,:pred_num,:,:3]

        #     vis_label = torch.any(cmp_3d != 0 , dim=-1, keepdim=True)
        #     cmp_3d = torch.cat([cmp_3d, vis_label.float()], dim = -1)
        #     _,_ = val_match(cmp_3d,gt_3d,mpjpe_orig[view])


    msg =  f'EPOCH {total_iter}: The MPJPE is {mpjpe.avg}, Precision {precision.avg}, Recall {recall.avg}'  #
    logger.info(msg)  
    for v in range(view_num):
        msg =  f'EPOCH {total_iter}: The MPJPE orig {v} is {mpjpe_orig[v].avg}, Precision {precision_orig[v].avg}, Recall {recall_orig[v].avg}'  #
        logger.info(msg)


def refine_2dgt(pred_2d, gt2d, threshold=50):
    # gt _2d + pred_depth + pred_visible
    # match the hip 2 hip can not be invisible
    pred_pos = pred_2d[:,:,:2].copy()
    pred_depth = pred_2d[:,:,2]
    pred_vis = pred_2d[:,:,3].copy()
    gt2d_pos = gt2d[:,:,:2].copy()
    gt2d_vis = gt2d[:,:,3].copy()
    gt_depth = gt2d[:,:,2]
    er_match = dict()
    er_id = dict()
    for idx, (pred, pred_v) in enumerate(zip(pred_pos, pred_vis)):
        temp_er = []
        mask1 = (pred_v > 0.3)
        for (gt, gt_v) in zip(gt2d_pos, gt2d_vis):
             # use the most confident joint
            mask2 = (gt_v > 0)
            mask = mask1 * mask2
            if np.sum(mask) == 0:
                temp_er.append(1000)
                continue
            c_err = np.mean(np.linalg.norm(pred[mask,:] - gt[mask,:], axis=-1))
            temp_er.append(c_err)
        min_gt = np.argmin(temp_er)
        min_er = np.min(temp_er)
        # consider the depth difference

        # mask1 = (pred_v > 0.3)
        # mpjpe = np.mean(np.abs(pred_depth[idx,mask] - gt_depth[min_gt,mask]))
        # if mpjpe > 10: # depth gap too much
        #     continue

        if min_gt in er_match.keys():
            if min_er < er_match[min_gt]:
                er_match[min_gt] = min_er
                er_id[min_gt] = idx
        else:
            er_match[min_gt] = min_er
            er_id[min_gt] = idx
    # delete the big err
    pred_idx = np.asarray(list(er_id.values()))
    if len(pred_idx) == 0:
        return None
    gt_idx = np.asarray(list(er_id.keys()))
    if len(gt_idx) == 0:
        return None
    valid_label = np.array(list(er_match.values())) < threshold 
    pred_idx = pred_idx[valid_label]
    gt_idx = gt_idx[valid_label]

    # replace the depth rather than the 2D coord
    c_pose = gt2d[gt_idx, :, :2]
    pred_pose = pred_2d[pred_idx,:,:].copy()
    pred_pose[:,:,:2] = c_pose
    # update the vis keys
    gt_vis = (gt2d[gt_idx, :, 3] != 0)
    pred_pose[:,:,3] = pred_pose[:,:,3] * gt_vis # keep vis consistent

    return pred_pose

def refine_2d_hrnet(pred_2d, hr2d, threshold=50):

    pred_pos = pred_2d[:,:,:2].copy()
    pred_vis = pred_2d[:,:,3].copy()
    gt2d_pos = hr2d[:,:,:2].copy()
    gt2d_vis = hr2d[:,:,2].copy()  # this vis is 0-1
    er_match = dict()
    er_id = dict()
    for idx, (pred, pred_v) in enumerate(zip(pred_pos, pred_vis)):
        temp_er = []
        mask1 = (pred_v > 0.3) # use the most confident joint
        for (gt, gt_v) in zip(gt2d_pos, gt2d_vis):
            mask2 = (gt_v > 0.1) # hrnet vis standard
            mask = mask1 * mask2
            if np.sum(mask) == 0:
                temp_er.append(1000)
                continue
            c_err = np.mean(np.linalg.norm(pred[mask,:] - gt[mask,:], axis=-1))
            temp_er.append(c_err)
        min_gt = np.argmin(temp_er)
        min_er = np.min(temp_er)
        # consider the depth difference

        # mask1 = (pred_v > 0.3)
        # mpjpe = np.mean(np.abs(pred_depth[idx,mask] - gt_depth[min_gt,mask]))
        # if mpjpe > 10: # depth gap too much
        #     continue

        if min_gt in er_match.keys():
            if min_er < er_match[min_gt]:
                er_match[min_gt] = min_er
                er_id[min_gt] = idx
        else:
            er_match[min_gt] = min_er
            er_id[min_gt] = idx
    # delete the big err
    pred_idx = np.asarray(list(er_id.values()))
    if len(pred_idx) == 0:
        return None
    gt_idx = np.asarray(list(er_id.keys()))
    if len(gt_idx) == 0:
        return None
    valid_label = np.array(list(er_match.values())) < threshold 
    pred_idx = pred_idx[valid_label]
    gt_idx = gt_idx[valid_label]
    
    c_pose = hr2d[gt_idx, :, :2]
    pred_pose = pred_2d[pred_idx,:,:].copy()
    pred_pose[:,:,:2] = c_pose
    # update the vis keys
    gt_vis = (hr2d[gt_idx, :, 2] > 0.1)
    pred_pose[:,:,3] = pred_pose[:,:,3] * gt_vis

    return pred_pose



def vis_joints3d(img_t, pred_3d, gt_3d, output_dir, idx, count, view_idx ,total_iter=0):# img_path,
    # img = cv2.imread(img_path)[:, :, ::-1]
    img = tensor2im(img_t)
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(141)
    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(142, projection='3d')
    for ip in range(len(pred_3d)):
        p3d = pred_3d[ip]
        for pair in pairs:
            # 存在
            ax2.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
    ax2.view_init(azim=-90, elev=-45)

    ax3 = fig.add_subplot(143, projection='3d')
    for ip in range(len(gt_3d)):
        p3d = gt_3d[ip]
        for pair in pairs:
            ax3.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)]) # linestyle=':'
    ax3.view_init(azim=-90, elev=-45)

    ax4 = fig.add_subplot(144, projection='3d')
    for ip in range(len(pred_3d)):
        p3d = pred_3d[ip]
        for pair in pairs:
            ax4.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)], linestyle= ':') # linestyle=':'
    for ip in range(len(gt_3d)):
        p3d = gt_3d[ip]
        for pair in pairs:
            ax4.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
    ax4.view_init(azim=-90, elev=-45)

    try:
        plt.savefig(os.path.join(output_dir,f'pose3d_iter_{total_iter}_i_{idx}_c_{count}_view_{view_idx}.jpg'))
    except:
        # print(img.shape)
        pass
    plt.close(fig)

def val(pred_3d, gt_3d, mpjpe, kp_set):
    pred_num = pred_3d.shape[0]
    pred_3d_vislabel = pred_3d[:,:,3]
    pred_3d = pred_3d[:,:,:3]
    # import pdb;pdb.set_trace()
    pred_3d_cn = pred_3d.reshape(pred_num,-1)
    pred_3d_cn = pred_3d_cn[:,None,:]
    actual_num = gt_3d.shape[0]
    pred_3d_cn = pred_3d_cn.repeat(1,actual_num,1)
    gt_3d_position = gt_3d[:,:,1:]
    gt_3d_cn = gt_3d_position.reshape(actual_num,-1) # long vector
    gt_valid = gt_3d[:,:,0]
    dist = torch.norm((pred_3d_cn - gt_3d_cn),dim=-1)
    match_idx = torch.argsort(dist, dim=-1)[:,0]
    for i in range(pred_num):
        match_i = match_idx[i] # 
        for j in range(cfg.DATASET.KEYPOINT.NUM):
            if gt_valid[match_i, j] !=0 and pred_3d_vislabel[i,j]!=0:
                error = torch.norm((gt_3d_position[match_i,j,:] - pred_3d[i,j,:])) # invisible joint process?
                kp_set[j].update(error.item())
                mpjpe.update(error.item())


def val_debug_bak(bak_pred, gt_3d, mpjpe, kp_set):
    # kp_set is in joint input
    # gt_3d : nposes x 3
    # bak_pred : 10 x 3
    nposes = gt_3d.shape[0]
    bak_num = bak_pred.shape[0]
    gt_3d_cn = gt_3d[:,None,:].repeat(1,bak_num,1)
    dist = torch.norm((gt_3d_cn - bak_pred),dim=-1)
    match_idx = torch.argsort(dist, dim=-1)[:,0]
    for i in range(nposes):
        error = torch.norm((gt_3d[i,:] - bak_pred[match_idx[i],:]))
        mpjpe.update(error.item())
        kp_set.update(error.item())

def val_debug(centers, masks, gt_3d, joint_idx, mpjpe, kp_set, fp_t, fp_k):
    # evaluatet the AP AR and MPJPE PCD PCK
    B = centers.shape[0]
    val_flag = 0
    for b in range(B): # divide in batch
        gt_3d_ex = gt_3d[b,...]
        nposes = torch.sum(gt_3d_ex[:,0,3]>0)
        gt_b = gt_3d_ex[:nposes,joint_idx,:3].cpu()
        pred_b = centers[b,...].cpu()
        mask_b = masks[b,...].cpu()
        pred_comp = pred_b[mask_b,...] # bool can just be in cpu
        pred_num = pred_comp.shape[0]
        # compare the corresponding kp within pred_comp and gt_b
        # process the AP and [mpjpe]
        gt_3d_cn = gt_b[:,None,:].repeat(1,pred_num,1)
        dist = torch.norm((gt_3d_cn - pred_comp),dim=-1)
        match_idx = torch.argsort(dist, dim=-1)[:,0]
        fp_t.update(abs(pred_num - nposes.item()), pred_num)
        fp_k.update(abs(pred_num - nposes.item()), pred_num)
        for i in range(nposes):
            error = torch.norm((gt_b[i,:] - pred_comp[match_idx[i],:]))
            if error.item() > 40:
                val_flag = 1
            mpjpe.update(error.item())
            kp_set.update(error.item())
    return val_flag, nposes

def val_fp(pred_3d_total, gt_3d, mpjpe, kp_set, threshold=500): # mainly FP
    pred_3d = pred_3d_total[...,:3]
    pred_vis = pred_3d_total[...,3]
    pred_num = pred_3d.shape[0]
    pred_3d_cn = pred_3d.reshape(pred_num,-1) # match the neck point
    pred_3d_cn = pred_3d_cn[:,None,:]
    actual_num = gt_3d.shape[0]
    pred_3d_cn = pred_3d_cn.repeat(1,actual_num,1)
    gt_3d_position = gt_3d[...,:3]
    gt_vis = gt_3d[...,3]
    gt_3d_cn = gt_3d_position.reshape(actual_num,-1) # long vector
    # gt_valid = gt_3d[:,:,0]
    dist = torch.norm((pred_3d_cn - gt_3d_cn),dim=-1)
    match_idx = torch.argsort(dist, dim=-1)[:,0]
    registor = AverageMeter() # reset per person
    vis_label = 0
    for i in range(pred_num):
        match_i = match_idx[i] # fp
        # perform the test
        for j in range(cfg.DATASET.KEYPOINT.NUM): #cfg.DATASET.KEYPOINT.NUM
            # if gt_valid[match_i, j] !=0 and pred_3d_vislabel[i,j]!=0:
            if gt_vis[match_i, j] >= 0.1 and pred_vis[i,j] != 0: # greater than 0.1
                error = torch.norm((gt_3d_position[match_i,j,:] - pred_3d[i,j,:])) # invisible joint process?
                registor.update(error.item())
                kp_set[j].update(error.item())
                # mpjpe.update(error.item())
        if registor.avg < threshold:
            mpjpe.update(registor.avg)
            if registor.avg > 40:
                vis_label = 1
        registor.reset()
    return vis_label

def val_fn(pred_3d_total, gt_3d, mpjpe, kp_set, threshold=500): # FN
    pred_3d = pred_3d_total[...,:3]
    pred_vis = pred_3d_total[...,3]
    pred_num = pred_3d.shape[0]
    pred_3d_cn = pred_3d.reshape(pred_num,-1) # match the neck point
    pred_3d_cn = pred_3d_cn[:,None,:]
    actual_num = gt_3d.shape[0]
    pred_3d_cn = pred_3d_cn.repeat(1,actual_num,1)
    gt_3d_position = gt_3d[...,:3]
    gt_vis = gt_3d[...,3]
    gt_3d_cn = gt_3d_position.reshape(actual_num,-1) # long vector
    # gt_valid = gt_3d[:,:,0]
    dist = torch.norm((pred_3d_cn - gt_3d_cn),dim=-1)
    dist = dist.t()
    match_idx = torch.argsort(dist, dim=-1)[:,0]
    registor = AverageMeter() # reset per person
    vis_label = 0
    for i in range(actual_num): # 
        match_i = match_idx[i] # 
        # perform the test
        for j in range(cfg.DATASET.KEYPOINT.NUM): #cfg.DATASET.KEYPOINT.NUM
            # if gt_valid[match_i, j] !=0 and pred_3d_vislabel[i,j]!=0:
            if gt_vis[i, j] >= 0.1 and pred_vis[match_i,j] != 0: # greater than 0.1
                error = torch.norm((gt_3d_position[i,j,:] - pred_3d[match_i,j,:])) # invisible joint process?
                registor.update(error.item())
                kp_set[j].update(error.item())
                # mpjpe.update(error.item())
        if registor.avg < threshold:
            mpjpe.update(registor.avg)
            if registor.avg > 40:
                vis_label = 1
        registor.reset()
    return vis_label    


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







def vis_2d(hmsIn_collect, img, folder_name, idx, idx_v, total_iter = 0): # root_d,
    # visualization
    # folder_name = os.path.join(cfg.OUTPUT_DIR, 'debug_test_pics')
    hm_folder = os.path.join(folder_name,'heatmap')
    # paf_folder = os.path.join(folder_name,'paf') # contains the
    # rootd_folder = os.path.join(folder_name,'rootdepth')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(hm_folder):
        os.makedirs(hm_folder)
    # if not os.path.exists(paf_folder):
    #     os.makedirs(paf_folder)
    # if not os.path.exists(rootd_folder):
    #     os.makedirs(rootd_folder)


    for hmsIn in hmsIn_collect:
        hmsIn = hmsIn.detach().cpu().numpy().transpose(1, 2, 0) # 直接输入
        hmsIn_upsamp = cv2.resize(
                        hmsIn, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
        vis_pic = tensor2im(img)
        # get the heatmap, paf, and relative depth map
        heatmap_vis = hmsIn_upsamp[...,:cfg.DATASET.KEYPOINT.NUM]
        # paf_vis = hmsIn_upsamp[...,cfg.DATASET.KEYPOINT.NUM:]
        
        # rdepth_vis = outputs['det_d'][-1][-1][0]

        # root depth did not exist the vis map, just show the final joint  verify the rgb
        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(vis_pic)
        # plt.subplot(122)
        # plt.imshow(root_d)
        # plt.savefig(os.path.join(rootd_folder, f'rdepth_{total_iter}_i_{idx}_v_{idx_v}.jpg'))
        # plt.close(fig)


        for hm_idx in range(cfg.DATASET.KEYPOINT.NUM):
            vis_hm = torch.tensor(heatmap_vis[...,hm_idx]).mul(255).clamp(0, 255).byte().numpy()
            colored_heatmap = cv2.applyColorMap(vis_hm, cv2.COLORMAP_JET)
            image_fusing = vis_pic * 0.3 + colored_heatmap * 0.7
            cv2.imwrite(os.path.join(hm_folder, f'hm_iter_{total_iter}_i_{idx}_v_{idx_v}_joints_{key_name[hm_idx]}.jpg'), image_fusing)

        # for con_idx in range(cfg.DATASET.PAF.NUM):
        #     # background = np.zeros((h,w))
        #     vis_paf = np.sum(np.abs(paf_vis[...,con_idx*2:con_idx*2+2]*127 ),axis=-1)
        #     vis_paf = torch.tensor(vis_paf).clamp(0, 255).byte().numpy()
        #     # .detach().cpu().numpy()

        #     colored_bool = cv2.applyColorMap(vis_paf, cv2.COLORMAP_JET)
        #     image_fusing = vis_pic * 0.3 + colored_bool * 0.7
        #     cv2.imwrite(os.path.join(paf_folder, f'paf_iter_{total_iter}_i_{idx}_v_{idx_v}_paf_{con_idx}.jpg'), image_fusing)    

def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = cfg.INPUT.MEANS #dataLoader中设置的mean参数
    std = cfg.INPUT.STDS  #dataLoader中设置的std参数
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): #如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.detach().cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): #反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
        temp = image_numpy[0,...].copy()
        image_numpy[0,...] = image_numpy[2,...].copy()
        image_numpy[2,...] = temp
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", "-t", type=str, default="run_inference",
                        choices=['generate_train', 'generate_result', 'run_inference', 'off_the_shelf','cross'],
                        help='Type of test. One of "generate_train": generate refineNet datasets, '
                             '"generate_result": save inference result and groundtruth, '
                             '"run_inference": save inference result for input images.')
    parser.add_argument("--data_mode", "-d", type=str, default="test",
                        choices=['test', 'generation'],
                        help='Only used for "generate_train" test_mode, "generation" for refineNet train dataset,'
                             '"test" for refineNet test dataset.')
    parser.add_argument("--SMAP_path", "-p", type=str, default='log/SMAP.pth',
                        help='Path to SMAP model')
    parser.add_argument("--RefineNet_path", "-rp", type=str, default='',
                        help='Path to RefineNet model, empty means without RefineNet')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch_size of test')
    parser.add_argument("--do_flip", type=float, default=0,
                        help='Set to 1 if do flip when test')
    parser.add_argument("--dataset_path", type=str, default="",
                        help='Image dir path of "run_inference" test mode')
    parser.add_argument("--json_name", type=str, default="",
                        help='Add a suffix to the result json.')
    args = parser.parse_args()
    cfg.TEST_MODE = args.test_mode
    cfg.DATA_MODE = args.data_mode
    cfg.REFINE = len(args.RefineNet_path) > 0
    cfg.DO_FLIP = args.do_flip
    cfg.JSON_SUFFIX_NAME = args.json_name
    cfg.TEST.IMG_PER_GPU = args.batch_size

    os.makedirs(cfg.TEST_DIR, exist_ok=True)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, 0, 'test_log_{}.txt'.format(args.test_mode))

    device = torch.device(cfg.MODEL.DEVICE)

    if args.test_mode == "off_the_shelf":
        # poserefine_file = '/home/panzhiyu/project/3d_pose/SMAP/model_logs_1019res/stage3_root2/iter-last.pth'
        poserefine_file = '/home/panzhiyu/project/3d_pose/SMAP/model_logs_1020_singletest/stage3_root2/best_model.pth'
        poserefine = Pose_GCN()
        poserefine.to(device) 
        state_dict = torch.load(poserefine_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        poserefine.load_state_dict(state_dict)
        pose_file = os.path.join('/home/panzhiyu/project/3d_pose/SMAP/model_logs_0821/stage3_root2/validation_result/','stage3_root2_generate_result_test_orig.pkl')
        easy_mode(pose_file, poserefine ,cfg, logger, device, output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result"))
    elif  args.test_mode == "cross":
        pose_file = os.path.join('/Extra/panzhiyu/CampusSeq1','smap_result_new.pkl')
        easy_mode_cross(pose_file, None ,cfg, logger, device, output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result"))

    else:
        model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)

        model.to(device)

        if args.test_mode == "run_inference":
            test_dataset = CustomDataset(cfg, args.dataset_path)
            data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            # print('use dataset!')
            data_loader = get_test_loader(cfg, num_gpu=1, local_rank=0, stage=args.data_mode)

        if cfg.REFINE:
            refine_model = RefineNet()
            refine_model.to(device)
            refine_model_file = args.RefineNet_path
        else:
            refine_model = None
            refine_model_file = ""

        model_file = args.SMAP_path
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            if os.path.exists(refine_model_file):
                refine_model.load_state_dict(torch.load(refine_model_file))
            elif refine_model is not None:
                logger.info("No such RefineNet checkpoint of {}".format(args.RefineNet_path))
                return
            generate_3d_point_pairs(model, refine_model, data_loader, cfg, logger, device,
                                    output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result")) #os.path.join(cfg.OUTPUT_DIR, "result")
        else:
            logger.info("No such checkpoint of SMAP {}".format(args.SMAP_path))


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
    main()
