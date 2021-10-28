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
from tqdm import tqdm
import numpy as np
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
namestr = ['CVLab1','CVLab2','CVLab3','CVLab4','IDIAP1','IDIAP2','IDIAP3']


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


class Wildtrack:
    def __init__(self, image_folder, keypoint_folder, view_set,is_train = True): # TODO add keypoint foldercfg,
        self.view_set = view_set
        # self.cam_list = [(0,x) for x in self.view_set] # get the HD images (0,)
        self.view_num = len(self.view_set)
        self.image_folder = image_folder
        # self.transform = transform
        self.conns = CONNS
        self.image_size = np.array([1920,1080])
        # self.input_size = cfg.dataset.input_size 
        # self.input_size = np.array([960,512])
        # self.heatmap_size = self.input_size / 2
        # self.heatmap_size = self.heatmap_size.astype(np.int16)

        self.num_joints = 15 #cfg.NETWORK.NUM_JOINTS
        self.paf_num = len(CONNS)

        # self.sigma = 4 #cfg.NETWORK.SIGMA
        # self.single_size = 512*424
        self.istrain = is_train
        # 读取k_calibration, ksync, 以depth 为准对齐一次即可
        # self.scene_num = len(self.scene_list)

        # calculate the total frame idx for the specific idx
        # read in the keypoint file as the order
        self.kpfiles = os.listdir(keypoint_folder)
        self.kpfiles.sort(key=lambda x:int(x.split('.')[0]))

        self.camera_parameters = self._get_cam()
        self.max_people = 40


    def _get_cam(self):
        cameras = {}
        for cam in self.view_set:
            sel_cam = {}
            root = ET.parse('/home/fbh/wildtrackdataset/Wildtrack_dataset/calibrations/extrinsic/extr_'+namestr[cam]+'.xml').getroot()
            R = root[0].text
            R = R.replace('\n', '')
            R = R.split(' ')
            Rm = []
            for str in R:
                if str!='':
                    Rm.append(float(str))
            rx = Rm[0]
            Rx = np.zeros([3,3])
            Rx[0,0] = 1
            Rx[1,1] = np.cos(rx)
            Rx[2,1] = np.sin(rx)
            Rx[1,2] = -np.sin(rx)
            Rx[2,2] = np.cos(rx)

            ry = Rm[1]
            Ry = np.zeros([3, 3])
            Ry[0, 0] = np.cos(ry)
            Ry[1, 1] = 1
            Ry[2, 0] = -np.sin(ry)
            Ry[0, 2] = np.sin(ry)
            Ry[2, 2] = np.cos(ry)

            rz = Rm[2]
            Rz = np.zeros([3, 3])
            Rz[0, 0] = np.cos(rz)
            Rz[0, 1] = -np.sin(rz)
            Rz[1, 0] = np.sin(rz)
            Rz[1, 1] = np.cos(rz)
            Rz[2, 2] = 1
            Rs = np.dot(np.dot(Rx,Ry),Rz)
            sel_cam['R'] = Rs
            # extr[:,0:3,i] = Rs
            t = root[1].text
            t = t.replace('\n', '')
            t = t.split(' ')
            tm = []
            for str in t:
                if str!='':
                    tm.append(float(str))
            Ts = np.stack(tm)
            sel_cam['t'] = Ts.reshape((3, 1))
            # extr[:,3,i] = np.stack(tm)
            sel_cam['distCoef'] = np.zeros(5)
            root = ET.parse('/home/fbh/wildtrackdataset/Wildtrack_dataset/calibrations/intrinsic_zero/intr_' + namestr[cam] + '.xml').getroot() # zero
            R = root[0][3].text
            R = R.replace('\n', '')
            R = R.split(' ')
            Rm = []
            for str in R:
                if str != '':
                    Rm.append(float(str))
            Rm = np.stack(Rm,axis = 0)
            Rm = np.reshape(Rm,[3,3])
            sel_cam['K'] = Rm
            # intr[:,:,i] = Rm


    def __len__(self):
        return int(np.sum(self.num_pers))

    def __generate_meta__(self):  # get the transfer (local to global) and the distosion parameters
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
            kp3d_body_data = kp_row_data['bodies']
            nposes = len(kp3d_body_data)
            if nposes == 0:
                continue
            number_people = torch.tensor(nposes)
            camera_paras = self.camera_parameters[scene_index]
            
            per_info_media = dict()
            for cam_node in self.cam_list:
                # generate meta file
                # different views
                
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
                Kd = view_para['distCoef'] # output this Kd for recovering 
                prefix = '{:02d}_{:02d}'.format(cam_node[0], cam_node[1])
                postfix = osp.basename(kp_file).replace('body3DScene', '')
                file_name = osp.join(self.image_folder,self.scene_list[scene_index],'hdImgs',prefix,
                                        prefix+postfix)
                file_name = file_name.replace('json','jpg')
                per_info['img_paths'] = file_name

                pose_3d = []
                pose_info = []
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

                    # # process the joints 
                    for i in range(len(pose2d)):
                        if (np.min(pose2d[i, :2]) < 0 or
                                pose2d[i, 0] >= self.image_size[0] or
                                pose2d[i, 1] >= self.image_size[1]):
                            anno_vis[i] = 0
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
                # meta['root'].append(per_info)
            meta['root'].append(per_info_media)
            
        if self.istrain:
            writen_file = osp.join(self.image_folder,'cmu_data_train_new5_multi.pkl')   
        else:
            writen_file = osp.join(self.image_folder,'cmu_data_test_alterview.pkl')
        
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
    img_path = '/Extra/fanbohao/fbh_data/wildtrackdataset/Wildtrack_dataset/Image_subsets/'
    kp_path = '/Extra/fanbohao/fbh_data/wildtrackdataset/Wildtrack_dataset/annotations_positions/'
    view_set = [0,1,2,3,4,5,6]

    depth_data_train = Wildtrack(img_path, kp_path, view_set,is_train = False)
    depth_data_train.__generate_meta__()



