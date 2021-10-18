# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import glob
import os.path as osp
import numpy as np
# import json_tricks as json 
import json
import pickle
import logging
import os 
import copy
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as transforms
import glob
import scipy.io as scio
from tqdm import tqdm

# train the campus

file = '/Extra/panzhiyu/CampusSeq1/calibration_campus.json'
# kpdata = scio.loadmat(file)
with open(file,'rb') as f:
    test_data = json.load(f)



TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    '160906_band3',
]
VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4']

CAMERA_NUMBER = (0,1,2)

CONNS =  [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]] # only 15 keypoints

Related = [14,13,12,6,7,8,11,10,9,3,4,5,0,1]

# '/Extra/panzhiyu/CMU_data/'
class Campus_Depth:
    def __init__(self, image_folder, view_set, istrain = True): # TODO add keypoint foldercfg,
        self.view_set = view_set
        # self.cam_list = [(0,x) for x in self.view_set] # get the HD images
        self.view_num = len(self.view_set)
        self.image_folder = image_folder
        # self.transform = transform
        self.conns = CONNS
        # self.image_size = np.array([360,288]) # Campus
        self.image_size = np.array([1032,776]) # Shelf

        self.num_joints = 15 #cfg.NETWORK.NUM_JOINTS
        self.paf_num = len(CONNS)

        # self.sigma = 4 #cfg.NETWORK.SIGMA
        # self.single_size = 512*424
        # self.istrain = is_train
        # if is_train:
        #     self.scene_list = TRAIN_LIST
        # else:
        #     self.scene_list = VAL_LIST
        # 读取k_calibration, ksync, 以depth 为准对齐一次即可
        # self.scene_num = len(self.scene_list)

        # self.calib_data = []
         

        # for scene in self.scene_list:
        #     with open(os.path.join(image_folder,scene,f'calibration_{scene}.json'),'rb') as dfile:
        #         self.calib_data.append(json.load(dfile))
        
        # calculate the total frame idx for the specific idx
        # read in the keypoint file as the order
        # self.kp3d_list = [osp.join(keypoint_folder, x, 'hdPose3d_stage1_coco19') for x in self.scene_list]
        self.kpfile = osp.join(image_folder, 'actorsGT.mat')

        self.kpdata = scio.loadmat(self.kpfile)
        self.kp3d = self.kpdata['actor3D'][0] # it has max people 3  
        self.dataset_len = len(self.kp3d[0]) # totally 2000 frames

        tris_list = list(range(self.dataset_len))

        # not used
        test_list = list(range(350,471))
        test_list.extend(list(range(650,751)))
        # get the camera parameters
        self.camera_parameters = (self._get_cam())
        
        self.max_people = 4 # campus 3 shelf 4

    def _get_cam(self):
        # cam_file = osp.join(self.image_folder, 'calibration_campus.json') # campus
        cam_file = osp.join(self.image_folder, 'calibration_shelf.json') # shelf
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        cameras = {}
        # for cam in calib['cameras']:
        #     if (cam['panel'], cam['node']) in self.view_set: # camera 位置信息的选择 （panel, node） 当前，视角就是Node决定
        #         sel_cam = {}
        #         sel_cam['K'] = np.array(cam['K'])
        #         sel_cam['distCoef'] = np.array(cam['distCoef'])
        #         sel_cam['R'] = np.array(cam['R']) #.dot(M)  # 旋转矩阵要处理一下 （坐标设置跟投影矩阵不匹配？）
        #         sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
        #         cameras[(cam['panel'], cam['node'])] = sel_cam
        for cam in self.view_set:
            sel_cam = {}
            sel_cam['R'] = np.array(calib[f'{cam}']['R'])
            sel_cam['t'] = np.array(calib[f'{cam}']['T']) 
            sel_cam['distCoef'] = np.zeros(5)
            sel_cam['K'] = np.array([[calib[f'{cam}']['fx'],0,calib[f'{cam}']['cx']],[0,calib[f'{cam}']['fy'], calib[f'{cam}']['cy']],[0,0,1]])
            # sel_cam['fx'] = np.array(calib[f'{cam}']['fx'])
            # sel_cam['fy'] = np.array(calib[f'{cam}']['fy'])
            # sel_cam['cx'] = np.array(calib[f'{cam}']['cx'])
            # sel_cam['cy'] = np.array(calib[f'{cam}']['cy'])
            cameras[cam] = sel_cam
        return cameras


    def __len__(self):
        return int(np.sum(self.num_pers)) # useless

    def __generate_meta__(self): 
        meta = dict()
        meta['root'] = list()

        for index in tqdm(range(self.dataset_len)):           
            # findtable = self.until_sum - (index + 1)
            # scene_index = np.min(np.where(findtable>=0)[0])
            # until_num = np.sum(self.num_pers[:scene_index])
            # kp_idx = index - until_num
            # kp_file = self.anno_kp_files[scene_index][kp_idx]
            
            # with open(kp_file,'rb') as kp: 
            #     try:
            #         kp_row_data = json.load(kp)
            #     except:
            #         continue
            # kp3d_body_data = kp_row_data['bodies']
            # nposes = len(kp3d_body_data)
            # if nposes == 0:
            #     continue
            # number_people = torch.tensor(nposes)
            # camera_paras = self.camera_parameters[scene_index]
            per_info_media = dict()
            for cam in self.view_set:
                # generate meta file
                # different views
                per_info = dict()
                per_info['img_height'] = int(self.image_size[1])
                per_info['img_width'] = int(self.image_size[0])
                per_info['dataset'] = 'SHELF' # for campus
                per_info['isValidation'] = 1
                
                # judge the 2D valid?
                view_para = self.camera_parameters[cam]
                K = view_para['K']
                R = view_para['R']
                T = view_para['t']
                Kd = view_para['distCoef']
                # D = np.concatenate([R,T],axis=-1)
                # P = K @ D
                pose_info = []
                pose_3d = []
                for n in range(self.max_people):
                    # extract the kp_info
                    extracted_pose = self.kp3d[n][index][0]    #[ped][frame][0]
                    if len(extracted_pose) < 14:
                        continue
                    # process the 3d pose into 15 joints and cmu form
                    pose3d = np.zeros((15,3),np.float32)
                    mid_hip = np.mean(np.concatenate([extracted_pose[2:3,:],extracted_pose[3:4,:]],axis=0),axis=0)
                    pose3d[2,:] = mid_hip
                    pose3d[Related,:] = extracted_pose

                    pose2d, depth_val, points_3d_cam, fx, fy, cx, cy = self.__projectjointsPoints__(pose3d.transpose(),K, R, T, Kd)
                    x_check = np.bitwise_and(pose2d[:, 0] >= 0, 
                                                pose2d[:, 0] <= self.image_size[0] - 1) #(15,) bool
                    y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                pose2d[:, 1] <= self.image_size[1] - 1)
                    check = np.bitwise_and(x_check, y_check)
                    valid_flag = np.sum(check)
                    if valid_flag < 3: # consider it do not has person in this view
                        continue
                    anno_vis = 2 * np.ones(15)
                    anno_vis[np.logical_not(check)] = 0
                    anno_vis = np.expand_dims(anno_vis,axis = -1)
                    joints_info = np.concatenate([pose2d, depth_val, anno_vis, points_3d_cam, fx, fy, cx, cy], axis=-1)
                    pose_info.append(joints_info[None,...])
                    pose_3d.append(pose3d[None,...])

                if len(pose_info) == 0:
                    continue
                pose_info = np.concatenate(pose_info,axis=0)
                pose_3d = np.concatenate(pose_3d, axis=0)
                per_info['bodys_3d'] = pose_3d
                per_info['bodys'] = pose_info
                # per_info['img_paths'] = osp.join(self.image_folder, f'Camera{cam}',f'campus4-c{cam}-{index:0>5d}.png')
                per_info['img_paths'] = osp.join(self.image_folder, f'Camera{cam}',f'img_{index:0>6d}.png')
                # meta['root'].append(per_info)
                per_info['cam'] = view_para
                per_info_media[cam] = per_info
            
            if len(per_info_media) < 5: # 3 views for campus and 5 views for shelf
                continue
            meta['root'].append(per_info_media)

                # prefix = '{:02d}_{:02d}'.format(cam_node[0], cam_node[1])
                # postfix = osp.basename(kp_file).replace('body3DScene', '')
                # file_name = osp.join(self.image_folder,self.scene_list[scene_index],'hdImgs',prefix,
                #                         prefix+postfix)
                # file_name = file_name.replace('json','jpg')
                # per_info['img_paths'] = file_name



                # pose_info = []
                # for n in range(nposes):
                #     pose3d = np.array(kp3d_body_data[n]['joints19']).reshape((-1, 4))
                #     # process the joint into 15 keypoints
                #     pose3d = pose3d[:15,:].copy() # only consider 15 keypoints
                #     anno_vis = 2 * (pose3d[:, -1] > 0.1)
                #     pose3d_proc = pose3d[...,:3].copy()
                #     pose2d, depth_val, points_3d_cam, fx, fy, cx, cy = self.__projectjointsPoints__(pose3d_proc.transpose(),K, R, T, Kd) # get the corresponding joint depth value

                #     x_check = np.bitwise_and(pose2d[:, 0] >= 0, 
                #                                 pose2d[:, 0] <= self.image_size[0] - 1) #(15,) bool
                #     y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                #                                 pose2d[:, 1] <= self.image_size[1] - 1)

                #     check = np.bitwise_and(x_check, y_check) # check bool se
                #     anno_vis[np.logical_not(check)] = 0


                #     anno_vis = np.expand_dims(anno_vis,axis = -1)
                #     joints_info = np.concatenate([pose2d, depth_val, anno_vis, points_3d_cam, fx, fy, cx, cy], axis=-1)
                #     pose_info.append(joints_info[None,...])

                # pose_info = np.concatenate(pose_info,axis=0)
                # per_info['bodys'] = pose_info
                # meta['root'].append(per_info)
        
        # if self.istrain:
            # writen_file = osp.join(self.image_folder,'cmu_data_train.pkl')   
        # else:
        writen_file = osp.join(self.image_folder,'shelf_meta_multi.pkl')
        
        with open(writen_file,'wb') as f:
            pickle.dump(meta, f)

    def __projectjointsPoints__(self, X, K, R, t, Kd):
        """
        Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        Roughly, x = K*(R*X + t) + distortion
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """

        x = (np.dot(R, X) + t)  # panoptic to kinect color scaling  cm to m 
        points_3d_cam = x.copy().transpose() #[J,3]
        depth_val = x[2:3,:].transpose() # the depth value of current joint, metrix is meter [N,1]

        x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

        r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

        # 去畸变
        x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                            ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                                r + 2 * x[0, :] * x[0, :])
        x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                            ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                                r + 2 * x[1, :] * x[1, :])

        x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
        x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

        # depth_val_norm = depth_val * W / f  # absolute depth sensing
        pose_2d = x[:2,:].copy().transpose()
        fx = K[0,0] * np.ones((pose_2d.shape[0],1))
        fy = K[1,1] * np.ones((pose_2d.shape[0],1))
        cx = K[0,2] * np.ones((pose_2d.shape[0],1))
        cy = K[1,2] * np.ones((pose_2d.shape[0],1))

        return pose_2d, depth_val, points_3d_cam, fx, fy, cx, cy

    

if __name__ == '__main__':
    # img_path = '/Extra/panzhiyu/CampusSeq1/'
    img_path = '/Extra/panzhiyu/Shelf/'
    view_set = [0,1,2,3,4]

    depth_data_train = Campus_Depth(img_path, view_set)
    # depth_data_test = Campus_Depth(img_path,view_set)
    depth_data_train.__generate_meta__()
    # depth_data_test.__generate_meta__()


