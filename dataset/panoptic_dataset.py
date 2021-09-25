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
from functools import reduce
import matplotlib.pyplot as plt
import glob

# GEN_LIST = [
# "160226_haggling1",
# "160906_ian1",
# "160906_ian2",
# "160906_band1",
# "160906_band2",
# "160906_pizza1",
# "160422_haggling1",
# "160906_ian5",
# ]
# TEST_LIST = [
#     "161202_haggling1",
#     "160906_ian3",
#     "160906_band3",
# ]

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

class Panoptic_Depth(Dataset):
    def __init__(self,  image_folder, keypoint_folder, view_set,is_train = True, transform = None): # TODO add keypoint foldercfg,
        self.view_set = view_set
        self.cam_list = [(0,x) for x in self.view_set] # get the HD images
        self.view_num = len(self.view_set)
        self.image_folder = image_folder
        self.transform = transform
        self.conns = CONNS
        # self.overall_view = [1,2,3,4,5]
        self.depth_size = np.array([512,424])
        self.image_size = np.array([1920,1080])
        # self.input_size = cfg.dataset.input_size 
        self.input_size = np.array([960,512])
        self.heatmap_size = self.input_size / 2
        self.heatmap_size = self.heatmap_size.astype(np.int16)

        self.num_joints = 15 #cfg.NETWORK.NUM_JOINTS
        self.paf_num = len(CONNS)

        self.sigma = 4 #cfg.NETWORK.SIGMA
        self.single_size = 512*424
        if is_train:
            self.scene_list = TRAIN_LIST
        else:
            self.scene_list = VAL_LIST
        # 读取k_calibration, ksync, 以depth 为准对齐一次即可
        self.scene_num = len(self.scene_list)

        self.calib_data = []
        # Using the HD images 

        for scene in self.scene_list:
            with open(os.path.join(image_folder,scene,f'calibration_{scene}.json'),'rb') as dfile:
                self.calib_data.append(json.load(dfile))
        
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

        # get the camera parameters
        self.camera_parameters = []
        for scene in self.scene_list:
            self.camera_parameters.append(self._get_cam(scene))
        
        self.max_people = 10

    def _get_cam(self, scene):
        cam_file = osp.join(self.image_folder, scene, 'calibration_{:s}.json'.format(scene)) 
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

    def __getitem__(self,index): 

        # return the original representation
        findtable = self.until_sum - (index + 1)
        scene_index = np.min(np.where(findtable>=0)[0])
        until_num = np.sum(self.num_pers[:scene_index])
        kp_idx = index - until_num
        kp_file = self.anno_kp_files[scene_index][kp_idx]
        with open(kp_file,'rb') as kp: 
            try:
                kp_row_data = json.load(kp)
            except:
                return None,None,None,None,None,None,None

        kp3d_body_data = kp_row_data['bodies']
        nposes = len(kp3d_body_data)
        if nposes == 0:
            return None, None, None, None, None,None,None

        number_people = torch.tensor(nposes)
        camera_paras = self.camera_parameters[scene_index]

        img_out = []
        pose2d_out = []
        hm_out = []
        paf_out = []
        weight_out = []
        jointdepth_out = []
        for cam_node in self.cam_list:
            view_para = camera_paras[cam_node]
            K = view_para['K']
            R = view_para['R']
            T = view_para['t']
            Kd = view_para['distCoef']
            prefix = '{:02d}_{:02d}'.format(cam_node[0], cam_node[1])
            postfix = osp.basename(kp_file).replace('body3DScene', '')
            file_name = osp.join(self.image_folder,self.scene_list[scene_index],'hdImgs',prefix,
                                    prefix+postfix)
            file_name = file_name.replace('json','jpg')
            img_frame = cv2.imread(file_name,
                                    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            # img_frame = cv2.cvtColor(img_frame,cv2.COLOR_BGR2RGB)

            # do the trans for the network input
            height, width, _ = img_frame.shape
            c = np.array([width / 2.0, height / 2.0]) # 
            s = self.__get_scale__((width, height), self.input_size) # (960 * 512) # 读入和实际
            r = 0

            trans = self.__get_affine_transform__(c, s, r, self.input_size)  # 2D GT 做同样的变换 # 表示平移和旋转关系

            img = cv2.warpAffine(
                    img_frame,
                    trans, (int(self.input_size[0]), int(self.input_size[1])),
                    flags=cv2.INTER_LINEAR)


            # do transform
            if self.transform:
                img = self.transform(img)

            img_out.append(img)  #.unsqueeze(0)

            # poses_3d = []
            # poses_vis_3d = []
            joints = []
            joints_vis = []
            joints_ = []
            # joints_depth = []
            for n in range(nposes):
                pose3d = np.array(kp3d_body_data[n]['joints19']).reshape((-1, 4))
                # process the joint into 15 keypoints
                pose3d = pose3d[:15,:].copy() # only consider 15 keypoints
                anno_vis = pose3d[:, -1] > 0.1
                pose3d_proc = pose3d[...,:3].copy()
                pose2d, pose_depth = self.__projectjointsPoints__(pose3d_proc.transpose(),K, R, T, Kd) # get the corresponding joint depth value

                pose2d = pose2d[:2,:].transpose()

                x_check = np.bitwise_and(pose2d[:, 0] >= 0, 
                                            pose2d[:, 0] <= self.image_size[0] - 1) #(15,) bool
                y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                            pose2d[:, 1] <= self.image_size[1] - 1)

                check = np.bitwise_and(x_check, y_check) # check bool se
                anno_vis[np.logical_not(check)] = 0

                # process the joints 
                for i in range(len(pose2d)):
                    pose2d[i, 0:2] = self.__affine_transform__(  # joints 为 GT 处理的结果 # transform on the pose2d
                        pose2d[i, 0:2], trans)
                    if (np.min(pose2d[i, :2]) < 0 or
                            pose2d[i, 0] >= self.input_size[0] or
                            pose2d[i, 1] >= self.input_size[1]):
                        anno_vis[i] = 0

                joints_depth.append(pose_depth)
                joints.append(pose2d) # pose after transform
                anno_vis = np.expand_dims(anno_vis,axis = -1)
                joints_.append(np.concatenate([pose2d, pose_depth, anno_vis],axis=-1)) # it has 4 dimensions
                joints_vis.append(anno_vis)

            # joints_ is the list contains the key info

            target_heatmap, gt_paf, joint_depthmap, target_weight = self.__generate_target_heatmap__(
                joints, joints_depth, joints_vis)  # changed the depth to the uniformed one

            # process the 2D joint information
            joints_depth = np.concatenate(joints_depth, axis=0)
            joints_depth = joints_depth.reshape(nposes, -1,1)
            joints_2D = np.concatenate(joints, axis=0)
            joints_2D = joints_2D.reshape(nposes,-1,2)
            
            joints_2d = np.concatenate([joints_2D, joints_depth], axis=-1)
            joints_2d = torch.from_numpy(joints_2d) # it is not lined

            offset = self.max_people - nposes
            fill_indx = torch.randint(nposes, (offset, ))
            fill_tensor = joints_2d[fill_indx,:]
            output_pose_2d = torch.cat([joints_2d,fill_tensor],dim=0)

            pose2d_out.append(output_pose_2d) #.unsqueeze(0)


            # generate the paf
            # gt_paf = self.__genPafs__(joints, joints_vis, self.conns)
            # do the crop  # We do not need to crop this heatmap
            # target_heatmap = target_heatmap[:,:,71:411]
            target_heatmap = torch.from_numpy(target_heatmap)
            hm_out.append(target_heatmap) #.unsqueeze(0)

            target_weight = torch.from_numpy(target_weight)
            weight_out.append(target_weight) #.unsqueeze(0)

            # joint_depthmap = joint_depthmap[:,:,71:411]
            joint_depthmap = torch.from_numpy(joint_depthmap)    ### psedo depth
            jointdepth_out.append(joint_depthmap) #.unsqueeze(0)
            # gt_paf = gt_paf[:,:,71:411] # do not need to crop the image
            gt_paf = torch.from_numpy(gt_paf)
            paf_out.append(gt_paf)  #.unsqueeze(0)

        # img_out = torch.cat(img_out, dim=0)
        # pose2d_out = torch.cat(pose2d_out, dim=0)
        # hm_out = torch.cat(hm_out,dim=0)
        # paf_out = torch.cat(paf_out, dim=0)
        # weight_out = torch.cat(weight_out,dim=0)
        # jointdepth_out = torch.cat(jointdepth_out,dim=0)

        return img_out, pose2d_out, hm_out, weight_out, paf_out, number_people, jointdepth_out

    def __affine_transform__(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

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
        depth_val = x[2:3,:].transpose() # shape (1, N)

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

        return x, depth_val

    def __get_scale__(self, image_size, resized_size): # confirm the equal_scale transform
        w, h = image_size
        w_resized, h_resized = resized_size  # no padding
        w_pad = w
        h_pad = h
        # if w / w_resized < h / h_resized:
        #     w_pad = h / h_resized * w_resized
        #     h_pad = h
        # else:
        #     w_pad = w
        #     h_pad = w / w_resized * h_resized
        scale = np.array([w_pad / 200.0, h_pad / 200.0], dtype=np.float32)

        return scale

    def __projectjointsPoints__(self, X, K, R, t, Kd):
        """
        Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        Roughly, x = K*(R*X + t) + distortion
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """

        x = (np.dot(R, X) + t)  # panoptic to kinect color scaling  cm to m 
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

        W = 1920
        f = K[0,0]
        # depth_val_norm = depth_val * W / f  # absolute depth sensing

        return x, depth_val


    def __get_affine_transform__(self, center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
        if isinstance(scale, torch.Tensor):
            scale = np.array(scale.cpu())
        if isinstance(center, torch.Tensor):
            center = np.array(center.cpu())
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale * 200.0
        src_w, src_h = scale_tmp[0], scale_tmp[1]
        dst_w, dst_h = output_size[0], output_size[1]

        rot_rad = np.pi * rot / 180
        if src_w >= src_h:
            src_dir = self.__get_dir__([0, src_w * -0.5], rot_rad)
            dst_dir = np.array([0, dst_w * -0.5], np.float32)
        else:
            src_dir = self.__get_dir__([src_h * -0.5, 0], rot_rad)
            dst_dir = np.array([dst_h * -0.5, 0], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift     # x,y
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.__get_3rd_point__(src[0, :], src[1, :])
        dst[2:, :] = self.__get_3rd_point__(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def __get_dir__(self,src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def __get_3rd_point__(self,a, b):
        direct = a - b
        return np.array(b) + np.array([-direct[1], direct[0]], dtype=np.float32)

    def __generate_target_heatmap__(self, joints, joints_depth,joints_vis, threshold=3): # for 5 pixels
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        nposes = len(joints)
        num_joints = self.num_joints
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        target = np.zeros(
            (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)
        
        target_depth = np.ones((num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                                dtype=np.float32) * 100

        pafs = np.zeros(
                (len(self.conns) * 2, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)

        feat_stride = self.input_size / self.heatmap_size

        for n in range(nposes):
            human_scale = 2 * self.__compute_human_scale__(joints[n] / feat_stride, joints_vis[n]) # TODO: compute human scale
            if human_scale == 0:
                continue
            # generate the heatmap
            # get the scale
            cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0))) # adjust according to the human scale in 2D 
            tmp_size = cur_sigma * 3
            tmp_size_depth = (cur_sigma/2) * 3
            for joint_id in range(num_joints):
                feat_stride = self.input_size / self.heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                # ul_d = [int(mu_x - tmp_size_depth), int(mu_y - tmp_size_depth)]
                # br_d = [int(mu_x + tmp_size_depth + 1), int(mu_y + tmp_size_depth + 1)]

                if joints_vis[n][joint_id, 0] == 0 or \
                        ul[0] >= self.heatmap_size[0] or \
                        ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(
                    -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                current_depth_value = joints_depth[n][joint_id] # get the current depth value of the joint

                size_depth = 2 * tmp_size_depth + 1
                x = np.arange(0, size_depth, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size_depth // 2
                # g_depth = current_depth_value * np.exp(
                #                                 -((x - x0)**2 + (y - y0)**2) / (2 * (cur_sigma/2)**2))
                # Usable gaussian range
                g_x = max(0,
                            -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0,
                            -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]

                # g_x_d = max(0,
                #             -ul_d[0]), min(br_d[0], self.heatmap_size[0]) - ul_d[0]
                # g_y_d = max(0,
                #             -ul_d[1]), min(br_d[1], self.heatmap_size[1]) - ul_d[1]

                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                # img_x_d = max(0, ul_d[0]), min(br_d[0], self.heatmap_size[0])
                # img_y_d = max(0, ul_d[1]), min(br_d[1], self.heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                depth_threshold_mask = (target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] > 0.7) # candidate region
                # changed to depth map representation  g_depth[g_y_d[0]:g_y_d[1], g_x_d[0]:g_x_d[1]]
                # target_depth[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]][depth_threshold_mask] = np.minimum(target_depth[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]][depth_threshold_mask],current_depth_value)
                target_depth[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]][depth_threshold_mask] = np.where(target_depth[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]][depth_threshold_mask]>current_depth_value, current_depth_value, target_depth[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]][depth_threshold_mask])
            target = np.clip(target, 0, 1)
            # generate the paf
            for (k, conn) in enumerate(self.conns):
                # feat_stride = self.input_size / self.heatmap_size
                pafa = pafs[k * 2,:, :]
                pafb = pafs[k * 2 + 1,:, :]
                points1 = joints[n][conn[0], :]
                points2 = joints[n][conn[1], :]
                x_center1 = points1[0] / feat_stride[0]
                x_center2 = points2[0] / feat_stride[0]
                y_center1 = points1[1] / feat_stride[1]
                y_center2 = points2[1] / feat_stride[1]

                line = np.array((x_center2 - x_center1, y_center2 - y_center1))
                if np.linalg.norm(line) == 0:
                    continue
                x_min = max(int(round(min(x_center1, x_center2) - threshold)), 0)
                x_max = min(int(round(max(x_center1, x_center2) + threshold)), self.heatmap_size[0])
                y_min = max(int(round(min(y_center1, y_center2) - threshold)), 0)
                y_max = min(int(round(max(y_center1, y_center2) + threshold)), self.heatmap_size[1])

                line /= np.linalg.norm(line)
                # vx, vy = [paf[y_min:y_max, x_min:x_max] for paf in (pafa, pafb)]
                xs = np.arange(x_min, x_max)
                ys = np.arange(y_min, y_max)[:, np.newaxis]

                v0, v1 = xs - x_center1, ys - y_center1
                dist = abs(v0 * line[1] - v1 * line[0])
                idxs = dist < threshold

                pafa[y_min:y_max, x_min:x_max][idxs] = line[0]
                pafb[y_min:y_max, x_min:x_max][idxs] = line[1]

        target_depth[target_depth == 100] = 0

        return target, pafs, target_depth ,target_weight

    def __compute_human_scale__ (self, pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])  # calculate the bounding box 
        # return np.clip((maxy - miny) * (maxx - minx), 1.0 / 4 * 256**2,
        #                4 * 256**2)
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 2 * 96**2, 4 * 96**2)    
    

if __name__ == '__main__':
    img_path = '/Extra/panzhiyu/CMU_kinect_data/'
    kp_path = '/Extra/panzhiyu/CMU_data/'
    view_set = [1,2,3]
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    depth_data = Panoptic_Depth(img_path, kp_path, view_set,is_train=True,
                                transform = transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ]))
    a,b = depth_data[1400]
    print('xx')

# 