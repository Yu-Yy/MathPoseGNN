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
from functools import reduce
import matplotlib.pyplot as plt 
from tqdm import tqdm

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
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18, # 恢复后四点
} # panoptic 实际使用关节点数为15


# TRAIN_LIST = [
# "160226_haggling1",
# "160906_ian1",
# "160906_ian2",
# "160906_band1",
# "160906_band2",
# "160906_pizza1",
# "160422_haggling1",
# "160906_ian5",
# ]

TRAIN_LIST = [
    '160226_haggling1',
    '160422_ultimatum1',
    '160224_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    '160906_band3',
]

# train the dataset in the 2D's test set
# GEN_LIST = ["161202_haggling1",
#             "160906_ian3", 
# ]
VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5'] #, '160906_band4'

# VAL_LIST = [   # Part of Val_list
#     "161202_haggling1",
#     "160906_ian3",
#     "160906_band3",
# ]

CAMERA_NUMBER = (3,6,12,13,23)

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

class Panoptic_Depth:
    def __init__(self, image_HD_folder, image_kinect_folder, keypoint_folder, view_set,is_train = True): # TODO add keypoint foldercfg,
        self.view_set = view_set
        self.kinect_viewset = list(range(1,6))
        self.cam_list = [(0,x) for x in self.view_set] # get the HD images (0,)
        self.view_num = len(self.view_set)
        self.image_HD_folder = image_HD_folder
        self.image_ki_folder = image_kinect_folder
        # self.transform = transform
        self.conns = CONNS
        self.image_size = np.array([1920,1080])
        self.depth_size = np.array([512,424])
        # self.input_size = cfg.dataset.input_size 
        # self.input_size = np.array([960,512])
        # self.heatmap_size = self.input_size / 2
        # self.heatmap_size = self.heatmap_size.astype(np.int16)

        self.num_joints = 15 #cfg.NETWORK.NUM_JOINTS
        self.paf_num = len(CONNS)

        # self.sigma = 4 #cfg.NETWORK.SIGMA
        self.single_size = 512*424
        self.istrain = is_train
        if is_train:
            self.scene_list = TRAIN_LIST
        else:
            self.scene_list = VAL_LIST
        # 读取k_calibration, ksync, 以depth 为准对齐一次即可
        self.scene_num = len(self.scene_list)
        # read the calibration file and the synctable
        self.kcalib_data = []
        self.ksync_data = []
        self.sync_data = []
        self.calib_data = []
        # Using the HD images 
        # use the kinect folder to load the data
        for scene in self.scene_list:
            with open(os.path.join(image_kinect_folder,scene,f'kcalibration_{scene}.json'),'rb') as dfile:
                self.kcalib_data.append(json.load(dfile))
        for scene in self.scene_list:
            with open(os.path.join(image_kinect_folder,scene,f'ksynctables_{scene}.json'),'rb') as dfile:
                self.ksync_data.append(json.load(dfile))
        for scene in self.scene_list:
            with open(os.path.join(image_kinect_folder,scene,f'calibration_{scene}.json'),'rb') as dfile:
                self.calib_data.append(json.load(dfile))
        for scene in self.scene_list:
            with open(os.path.join(image_kinect_folder,scene,f'synctables_{scene}.json'),'rb') as dfile:
                self.sync_data.append(json.load(dfile))
        # calculate the total frame idx for the specific idx
        # read in the keypoint file as the order
        self.kp3d_list = [osp.join(keypoint_folder, x, 'hdPose3d_stage1_coco19') for x in self.scene_list]
        self.anno_kp_files = []
        self.num_pers = []
        for kp_anno in self.kp3d_list:
            self.anno_kp_files.append(sorted(glob.iglob('{:s}/*.json'.format(kp_anno)))) 
            self.num_pers.append(len(self.anno_kp_files[-1]))

        # self.anno_kp_files = np.array(self.anno_kp_files)

        self.num_pers = np.array(self.num_pers)
        self.until_sum = np.cumsum(self.num_pers)

        # get the HD camera parameters
        self.HDcamera_parameters = []
        for scene in self.scene_list:
            self.HDcamera_parameters.append(self._get_cam(scene))
        
        self.max_people = 10
        self.dataset_len = int(np.sum(self.num_pers))

        self.scale_kinoptic2panoptic = np.eye(4)
        scaleFactor = 100
        self.scale_kinoptic2panoptic[:3,:3] = scaleFactor * self. scale_kinoptic2panoptic[:3,:3]
    
    def _get_cam(self, scene):
        cam_file = osp.join(self.image_HD_folder, scene, 'calibration_{:s}.json'.format(scene))  # It is actually not related to the scene
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list: # camera 位置信息的选择 （panel, node） 当前，视角就是Node决定
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']) #.dot(M)  # 旋转矩阵要处理一下 （坐标设置跟投影矩阵不匹配？）
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def __len__(self):
        return int(np.sum(self.num_pers))

    def __get_label__(self):
        meta = dict()
        meta['root'] = list()

        for index in tqdm(range(self.dataset_len)):
            findtable = self.until_sum - (index + 1)
            scene_index = np.min(np.where(findtable>=0)[0])
            until_num = np.sum(self.num_pers[:scene_index])
            kp_idx = index - until_num
            kp_file = self.anno_kp_files[scene_index][kp_idx]

            with open(kp_file,'rb') as kp: 
                try:
                    kp_row_data = json.load(kp)
                except:
                    continue

            univ_time = kp_row_data['univTime'] 
            kp3d_body_data = kp_row_data['bodies']
            nposes = len(kp3d_body_data)
            if nposes == 0:
                continue
            camera_paras = self.HDcamera_parameters[scene_index]
            # read the kinect depth data and generate the pointcloud
            points_collect = []
            for k_view in self.kinect_viewset:
                match_synctable_depth = np.array(self.ksync_data[scene_index]['kinect']['depth'][f'KINECTNODE{int(k_view)}']['univ_time'])
                target_frame_depth_idx = np.argmin(abs(match_synctable_depth - univ_time),axis=0)

                # get the kinect cam para

                panoptic_calibData = self.calib_data[scene_index]['cameras'][k_view + 509] # from 510 to 519
                M = np.concatenate([np.array(panoptic_calibData['R']),np.array(panoptic_calibData['t'])],axis=-1)
                T_panopticWorld2KinectColor = np.concatenate([M,np.array([[0,0,0,1]])],axis=0)
                T_kinectColor2PanopticWorld = np.linalg.pinv(T_panopticWorld2KinectColor) # Color is the camera coordinates
                kcalibdata = self.kcalib_data[scene_index]['sensors'][k_view - 1]
                T_kinectColor2KinectLocal = np.array(kcalibdata['M_color']) # Local is the kinect global coordinates
                T_kinectLocal2KinectColor = np.linalg.pinv(T_kinectColor2KinectLocal)
                T_kinectLocal2PanopticWorld = T_kinectColor2PanopticWorld @ self.scale_kinoptic2panoptic @ T_kinectLocal2KinectColor
                # T_kinectLocal2PanopticWorld = np.expand_dims(T_kinectLocal2PanopticWorld, axis = 0)

                # process the depth data
                fdepth = open(os.path.join(self.image_ki_folder,self.scene_list[scene_index],'kinect_shared_depth',f'KINECTNODE{int(k_view)}','depthdata.dat'),'rb')
                fdepth.seek(2*self.single_size*(target_frame_depth_idx), os.SEEK_SET)
                depth_org = np.fromfile(fdepth, count = self.single_size, dtype=np.int16)
                depth_org = depth_org.reshape([self.depth_size[1],self.depth_size[0]])
                depth_org = depth_org[...,::-1] * 0.001 # change in meter 
                depth_proc = depth_org.reshape(-1,1)
                point_3D, point_3d = self.__unprojectPoints__(kcalibdata, depth_org)

                point_global_3D = T_kinectLocal2PanopticWorld @ point_3D # metric is cm
                points_3d = point_global_3D[:3,:].transpose()
                points_collect.append(points_3d)
            pointcloud = np.concatenate(points_collect, axis=0)
            # get the col label
            per_info_media = dict()
            for cam_node in self.cam_list:

                per_info = dict()
                per_info['img_height'] = int(self.image_size[1])
                per_info['img_width'] = int(self.image_size[0])
                per_info['dataset'] = 'CMUP'
                if self.istrain:
                    per_info['isValidation'] = 0
                else:
                    per_info['isValidation'] = 1

                view_para = camera_paras[cam_node]
                K = view_para['K']
                R = view_para['R']
                T = view_para['t']
                Kd = view_para['distCoef']
                prefix = '{:02d}_{:02d}'.format(cam_node[0], cam_node[1])
                postfix = osp.basename(kp_file).replace('body3DScene', '')
                file_name = osp.join(self.image_HD_folder,self.scene_list[scene_index],'hdImgs',prefix,
                                        prefix+postfix)
                file_name = file_name.replace('json','jpg')
                per_info['img_paths'] = file_name

                pose_3d = []
                pose_info = []

                point2d, depth_val = self.__projectPoints__(pointcloud.transpose(),K, R, T, Kd)
                mask_depth = (depth_val.squeeze(-1) > 0)

                x_check = np.bitwise_and(point2d[:, 0] >= 0, 
                                            point2d[:, 0] <= self.image_size[0] - 1) #(15,) bool
                y_check = np.bitwise_and(point2d[:, 1] >= 0,
                                            point2d[:, 1] <= self.image_size[1] - 1)
                mask_2d = np.bitwise_and(x_check, y_check)
                mask = mask_depth * mask_2d
                point2d = point2d[mask,:]
                depth_c = 2000 * np.ones([self.image_size[1],self.image_size[0]])
                p_2d = point2d[:,:2]
                p_depth = point2d[:,2]
                sort_idx = np.argsort(p_depth)[::-1]
                p_2d = p_2d[sort_idx,:] 
                p_depth = p_depth[sort_idx]
                color_map = p_2d.astype(np.int16)
                # color_map[:,0] = np.clip(color_map[:,0], 0, self.image_size[0]-1)
                # color_map[:,1] = np.clip(color_map[:,1], 0, self.image_size[1]-1)
                depth_c[color_map[:,1],color_map[:,0]] = p_depth # fill the depth in other way
                for n in range(nposes):
                    pose3d = np.array(kp3d_body_data[n]['joints19']).reshape((-1, 4))
                    # process the joint into 15 keypoints
                    pose3d = pose3d[:15,:].copy() # only consider 15 keypoints # only 15 keypoints
                    anno_vis = 2 * (pose3d[:, -1] > 0.1)
                    pose3d_proc = pose3d[...,:3].copy()
                    pose2d, depth_val, points_3d_cam, fx, fy, cx, cy = self.__projectjointsPoints__(pose3d_proc.transpose(),K, R, T, Kd) # get the corresponding joint depth value

                    x_check = np.bitwise_and(pose2d[:, 0] >= 0, 
                                                pose2d[:, 0] <= self.image_size[0] - 1) #(15,) bool
                    y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                pose2d[:, 1] <= self.image_size[1] - 1)

                    check = np.bitwise_and(x_check, y_check) # check bool se
                    anno_vis[np.logical_not(check)] = 0
                    #  process the collision joints 
                    for i in range(len(pose2d)):
                        if anno_vis[i] > 0:
                            x_range = np.arange(pose2d[i, 0]-4,pose2d[i, 0]+4).astype(np.int16)
                            x_range = np.clip(x_range, 0, self.image_size[0] - 1)
                            y_range = np.arange(pose2d[i, 1]-4,pose2d[i, 1]+4).astype(np.int16)
                            y_range = np.clip(y_range, 0, self.image_size[1] - 1)
                            depth_ex = depth_c[y_range[0]:y_range[-1], x_range[0]:x_range[-1]]
                            depth_ex = depth_ex[depth_ex != 2000] # delete the blank point
                            # depth_c1 = np.tile(depth_ex[None,:], [num,1])
                            # depth_c2 = np.tile(depth_ex[:,None],).repeat(num)
                            if len(depth_ex) == 0:
                                continue
                            num = depth_ex.shape[0]
                            depth_ex = np.sort(depth_ex)
                            dist = depth_ex - depth_val[i]
                            jd = np.abs(dist) < 30
                            cons = np.any(jd)
                            if cons:
                                idx_judge = np.argmax(jd)
                                if (idx_judge+1) > num * 0.3:
                                    anno_vis[i] = 1                     
                            # con2 = 

                            # depth_comp_big = np.percentile(depth_ex, 95) 
                            # using the median value
                            # if depth_comp - depth_val[i] < -60: # consider it is collision
                            #     anno_vis[i] = 1 # set the collision flag
                    correct = anno_vis[anno_vis>0]
                    c_num = correct.shape[0]
                    if np.sum(correct == 2) > c_num * 0.8:
                        anno_vis[anno_vis==1] = 2
                    if np.sum(correct == 1) > c_num * 0.9:
                        anno_vis[anno_vis==2] = 1
                    anno_vis = np.expand_dims(anno_vis,axis = -1)
                    joints_info = np.concatenate([pose2d, depth_val, anno_vis, points_3d_cam, fx, fy, cx, cy], axis=-1) # the joint info
                    pose_info.append(joints_info[None,...])
                    pose_3d.append(pose3d[None,...])
                    
                pose_3d = np.concatenate(pose_3d, axis=0)
                per_info['bodys_3d'] = pose_3d
                per_info['cam'] = view_para # contains K R T Kd
                pose_info = np.concatenate(pose_info,axis=0)
                per_info['bodys'] = pose_info
                per_info_media[cam_node] = per_info

            meta['root'].append(per_info_media)
        
        if self.istrain:
            writen_file = osp.join(self.image_HD_folder,'cmu_data_train_multi_coll.pkl')   
        else:
            writen_file = osp.join(self.image_HD_folder,'cmu_data_test_multi_coll.pkl')
        
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



    def __unprojectPoints__(self,kcalibdata,depth_org):
        x_cor, y_cor = np.meshgrid(range(self.depth_size[0]), range(self.depth_size[1]))
        x_cor = x_cor.reshape(-1,1)
        y_cor = y_cor.reshape(-1,1)
        cor_2d = np.concatenate([x_cor,y_cor,np.ones([x_cor.shape[0],1])],axis=-1).transpose()
        K_depth = np.array(kcalibdata['K_depth'])
        norm_2d = np.linalg.pinv(K_depth) @ cor_2d
        norm_2d = norm_2d.transpose()
        x_cor_depth = norm_2d[:,0:1]
        x_cor_bak = x_cor_depth.copy()
        y_cor_depth = norm_2d[:,1:2]
        y_cor_bak = y_cor_depth.copy()
        K_diff = np.zeros(12)
        temp = np.array(kcalibdata['distCoeffs_depth'])
        K_diff[:5] = temp[:5].copy()
        # undistortion
        for _ in range(5):
            r2 = x_cor_depth * x_cor_depth + y_cor_depth * y_cor_depth
            icdist = (1 + ((K_diff[7]*r2 + K_diff[6])*r2 + K_diff[5])*r2) / (1 + ((K_diff[4]*r2 + K_diff[1])*r2 + K_diff[0])*r2)
            deltaX = 2*K_diff[2] *x_cor_depth *y_cor_depth + K_diff[3]*(r2 + 2*x_cor_depth * x_cor_depth)+  K_diff[8]*r2+K_diff[9]* r2 *r2
            deltaY = K_diff[2]*(r2 + 2*y_cor_depth *y_cor_depth) + 2*K_diff[3]*x_cor_depth *y_cor_depth+ K_diff[10] * r2 + K_diff[11]* r2 *r2

            x_cor_depth = (x_cor_bak - deltaX) *icdist
            y_cor_depth = (y_cor_bak - deltaY) *icdist

        depth_proc = depth_org.reshape(-1,1) 
        x_cor_depth = x_cor_depth * depth_proc
        y_cor_depth = y_cor_depth * depth_proc
        depth_cam = np.concatenate([x_cor_depth,y_cor_depth,depth_proc,np.ones(x_cor_depth.shape)],axis=-1)
        M_depth = np.array(kcalibdata['M_depth'])
        point_3D = np.linalg.pinv(M_depth) @ depth_cam.transpose()
        point_3d = point_3D[:3,:].transpose()

        return point_3D, point_3d

    def __projectPoints__(self, X, K, R, t, Kd):
        """
        Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        Roughly, x = K*(R*X + t) + distortion
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """

        x = np.dot(R, X) + t  # panoptic to kinect color scaling
        depth_val = x[2:3,:].transpose()

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
        
        x = x.transpose()
        return x, depth_val

if __name__ == '__main__':
    img_hd_path = '/Extra/panzhiyu/CMU_data/'
    img_ki_path = '/Extra/panzhiyu/CMU_kinect_data/'
    kp_path = '/Extra/panzhiyu/CMU_data/'
    view_set = [3,6,12,13,23]

    depth_data_train = Panoptic_Depth(img_hd_path, img_ki_path, kp_path, view_set,is_train = True)
    depth_data_test = Panoptic_Depth(img_hd_path, img_ki_path, kp_path, view_set,is_train = False)
    depth_data_train.__get_label__()
    depth_data_test.__get_label__()