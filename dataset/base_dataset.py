"""
@author: Jianan Zhen
@contact: jnzhen99@163.com
"""
import copy
import cv2
import json
import numpy as np
import os.path as osp
import torch
import pickle
from torch.utils.data import Dataset
from exps.stage3_root2.config import cfg
from dataset.ImageAugmentation import (aug_croppad, aug_flip, aug_rotate, aug_scale)
from dataset.representation import generate_heatmap, generate_paf, generate_rdepth


cam_nodes = cfg.DATASET.CAM #[(0,3),(0,6),(0,12),(0,13),(0,23)] # TODO: campus
# cam_nodes = [0,1,2]

class JointDataset(Dataset):
    def __init__(self, cfg, stage, transform=None, with_augmentation=False, with_mds=False):
        self.stage = stage
        """
        train: provide training data for training
        test: provide test data for test
        generation: provide training data for inference --> the input to RefineNet
        """
        assert self.stage in ('train', 'test', 'generation')

        self.transform = transform

        self.train_data = list()
        self.val_data = list()
        DATASET = cfg.dataset
        if self.stage == 'train':
            # choose coco + specific 3d dataset for training together
            # with open(DATASET.COCO_JSON_PATH) as data_file:
            #     data_this = json.load(data_file)
            #     data = data_this['root']
            # for data_name in DATASET.USED_3D_DATASETS: # 'MUCO', 'CMUP', 'H36M'
            with open(eval('DATASET.CMUP_JSON_PATH'),'rb') as data_file:  #CMUP
                data_this = pickle.load(data_file)
                data = data_this['root'] #+# load kp points
        elif self.stage == 'generation':
            data = []
            # # for data_name in DATASET.USED_3D_DATASETS: 
            #     with open(eval('DATASET.%s_JSON_PATH'%(data_name))) as data_file:
            #         data_this = json.load(data_file)
            #         data = data_this['root'] + data
            with open(eval('DATASET.CMUP_JSON_PATH'),'rb') as data_file:
                data_this = pickle.load(data_file)
                data = data_this['root'] + data

        else:
            with open(cfg.TEST.JSON_PATH, 'rb') as data_file:
                data_this = pickle.load(data_file)
                data = data_this['root']
        # different index may have different viewsW
        for i in range(len(data)): # transverse the whole image  # TODO: For campus demo
            # transverse all the views
            # self.val_data.append(data[i]) # TODO: for run the result of shelf or on cmu gnn # !!!!!
            if data[i][cam_nodes[0]]['isValidation'] != 0:
                self.val_data.append(data[i])
            else:
                self.train_data.append(data[i])

        self.input_shape = DATASET.INPUT_SHAPE
        self.output_shape = DATASET.OUTPUT_SHAPE
        self.stride = DATASET.STRIDE

        # data root path
        self.test_root_path = cfg.TEST.ROOT_PATH
        self.root_path = {}
        # for dname in (['COCO'] + DATASET.USED_3D_DATASETS): # 'MUCO', 'CMUP', 'H36M'

        self.root_path['CMUP'] = eval('DATASET.CMUP_ROOT_PATH') # TODO: CMUP

        # keypoints information
        self.root_idx = DATASET.ROOT_IDX
        self.keypoint_num = DATASET.KEYPOINT.NUM
        self.gaussian_kernels = DATASET.TRAIN.GAUSSIAN_KERNELS
        self.paf_num = DATASET.PAF.NUM
        self.paf_vector = DATASET.PAF.VECTOR
        self.paf_thre = DATASET.PAF.LINE_WIDTH_THRE

        # augmentation information
        self.with_augmentation = with_augmentation
        self.params_transform = dict()
        self.params_transform['crop_size_x'] = DATASET.INPUT_SHAPE[1]
        self.params_transform['crop_size_y'] = DATASET.INPUT_SHAPE[0]
        self.params_transform['center_perterb_max'] = DATASET.TRAIN.CENTER_TRANS_MAX
        self.params_transform['max_rotate_degree'] = DATASET.TRAIN.ROTATE_MAX
        self.params_transform['flip_prob'] = DATASET.TRAIN.FLIP_PROB
        self.params_transform['flip_order'] = DATASET.KEYPOINT.FLIP_ORDER
        self.params_transform['stride'] = DATASET.STRIDE
        self.params_transform['scale_max'] = DATASET.TRAIN.SCALE_MAX
        self.params_transform['scale_min'] = DATASET.TRAIN.SCALE_MIN

        self.with_mds = with_mds
        self.max_people = cfg.DATASET.MAX_PEOPLE # TODO: campus

    def __len__(self):
        if self.stage == 'train' or self.stage == 'generation':
            return len(self.train_data)
        else:  # 'test'
            return len(self.val_data)

    def get_anno(self, meta_data): # changed into multi-view
        anno = dict()
        for node in cam_nodes:
            # anno[node] = dict()
            # current_view = list(meta_data.keys())
            # if node not in current_view:
            #     continue
            anno_v = dict()
            anno_v['dataset'] = meta_data[node]['dataset'].upper()
            anno_v['img_height'] = int(meta_data[node]['img_height'])
            anno_v['img_width'] = int(meta_data[node]['img_width'])
            anno_v['img_paths'] = meta_data[node]['img_paths']
            anno_v['isValidation'] = meta_data[node]['isValidation']
            anno_v['bodys'] = np.asarray(meta_data[node]['bodys'])
            anno_v['center'] = np.array([anno_v['img_width']//2, anno_v['img_height']//2])
            anno_v['cam'] = meta_data[node]['cam']
            anno_v['bodys_3d'] = meta_data[node]['bodys_3d']
            anno[node] = anno_v

        return anno

    def remove_illegal_joint(self, meta):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        for i in range(len(meta['bodys'])):
            mask_ = np.logical_or.reduce((meta['bodys'][i][:, 0] >= crop_x, # 0,1代表2D值
                                          meta['bodys'][i][:, 0] < 0,
                                          meta['bodys'][i][:, 1] >= crop_y,
                                          meta['bodys'][i][:, 1] < 0))

            meta['bodys'][i][mask_ == True, 3] = 0 # 3为不可见 label 不可见部分设置为0， 2为zdepth
        return meta

    def __getitem__(self, index):
        if self.stage == 'train' or self.stage == 'generation':
            data = copy.deepcopy(self.train_data[index])
        else:
            data = copy.deepcopy(self.val_data[index])
        
        # generate the meta data
        meta_data_multi = self.get_anno(data)
        # current_views = list(meta_data_multi.keys())
        img_mul = []
        valid_mul = []
        labels_mul = []
        labels_rdepth_mul = []
        joints3d_local = []
        scale_mul = []
        img_paths = []
        joints3d = meta_data_multi[cam_nodes[0]]['bodys_3d']

        joints3d_align = np.zeros((self.max_people, self.keypoint_num, 4), np.float) 
        # joints3d_align = np.zeros((self.max_people, self.keypoint_num, 3), np.float) # TODO: Demo

        nposes = len(joints3d)
        joints3d_align[:nposes,...] = joints3d
        joints3d_align = torch.from_numpy(joints3d_align)
        cam_parameters_multi = []
        # this data is in multi_view
        for node in cam_nodes:
            meta_data = meta_data_multi[node]
            cam_parameters = meta_data['cam']
            cam_parameters_multi.append(cam_parameters)
            if self.stage not in ['train', 'generation']:
                root_path = self.test_root_path
            else:
                root_path = self.root_path[meta_data['dataset']]

            bodys = np.zeros((self.max_people, self.keypoint_num, len(meta_data['bodys'][0][0])), np.float)
            bodys[:len(meta_data['bodys'])] = np.asarray(meta_data['bodys'])
            joints3d_local.append(bodys[None,...])
            
            # read in the image
            # import pdb; pdb.set_trace()
            img = cv2.imread(osp.join(root_path,meta_data['img_paths']), cv2.IMREAD_COLOR) # TODO: campus 
            img_paths.append(meta_data['img_paths'])

            if self.with_augmentation:
                meta_data, img = aug_rotate(meta_data, img, self.params_transform)
            else:
                self.params_transform['center_perterb_max'] = 0
            
            # transform the image 数据增

            if meta_data['dataset'] == 'COCO':
                meta_data, img = aug_croppad(meta_data, img, self.params_transform, self.with_augmentation)
            else:
                meta_data, img = aug_croppad(meta_data, img, self.params_transform, False)

            if self.with_augmentation:
                meta_data, img = aug_flip(meta_data, img, self.params_transform)

            meta_data = self.remove_illegal_joint(meta_data)

            if self.transform:
                img = self.transform(img)
            else:
                img = img.transpose((2, 0, 1)).astype(np.float32)
                img = torch.from_numpy(img).float()  # generate the img 
        
            # generate labels
            valid = np.ones((self.keypoint_num +self.paf_num*3, 1), np.float)  
            if meta_data['dataset'] == 'COCO':
                # coco has no headtop annotation
                valid[1, 0] = 0  
                # pafs of headtop and neck
                valid[self.keypoint_num, 0] = 0   
                valid[self.keypoint_num+1, 0] = 0
                # relative depth
                valid[self.keypoint_num + self.paf_num*2:, 0] = 0


            if self.stage in ['train', 'generation']: # train then generate
                labels_num = len(self.gaussian_kernels) # labels_num is the person's number?
                labels = np.zeros((labels_num, self.keypoint_num + self.paf_num*3, *self.output_shape))  # TODO: The use of different guassian kernel
                for i in range(labels_num): # 不同 kernel size 下
                    # heatmaps
                    labels[i][:self.keypoint_num] = generate_heatmap(meta_data['bodys'], self.output_shape, self.stride, \
                                                                    self.keypoint_num, kernel=self.gaussian_kernels[i])  # heatmap channel
                    # pafs + relative depth
                    labels[i][self.keypoint_num:] = generate_paf(meta_data['bodys'], self.output_shape, self.params_transform, \
                                                                self.paf_num, self.paf_vector, max(1, (3-i))*self.paf_thre, self.with_mds)
                
                # root depth
                labels_rdepth = generate_rdepth(meta_data, self.stride, self.root_idx, self.max_people)
                labels = torch.from_numpy(labels).float()
                labels_rdepth = torch.from_numpy(labels_rdepth).float()
                valid = torch.from_numpy(valid).float()
                valid_mul.append(valid[None,...])
                labels_mul.append(labels[None,...])
                labels_rdepth_mul.append(labels_rdepth[None,...])

            
            scale_mul.append(meta_data['scale']) # different scale
            img_mul.append(img[None,...])
            

        img_mul = torch.cat(img_mul, dim=0)
        joints3d_local = np.concatenate(joints3d_local, axis=0)
        joints3d_local = torch.from_numpy(joints3d_local)

        if self.stage in ['test', 'generation']:
            # bodys = np.zeros((self.max_people, self.keypoint_num, len(meta_data['bodys'][0][0])), np.float)
            # bodys[:len(meta_data['bodys'])] = np.asarray(meta_data['bodys'])
            # img_path = data['img_paths']
            return img_mul, cam_parameters_multi, joints3d_align, joints3d_local , {'scale': scale_mul,  #meta_data['scale'],
                                                                    'nposes': nposes,
                                                                    'img_paths': img_paths,
                                                                    'img_width': meta_data['img_width'],
                                                                    'img_height': meta_data['img_height'],
                                                                    'net_width': self.params_transform['crop_size_x'],
                                                                    'net_height': self.params_transform['crop_size_y']}

        valid_mul = torch.cat(valid_mul,dim=0)
        labels_mul = torch.cat(labels_mul,dim=0)
        labels_rdepth_mul = torch.cat(labels_rdepth_mul,dim = 0)
        
        return img_mul, valid_mul, labels_mul, \
            labels_rdepth_mul, cam_parameters_multi, joints3d_align, joints3d_local ,{'scale': scale_mul, 
                                                                    'nposes': nposes,
                                                                    'img_paths': img_paths,
                                                                    'img_width': meta_data['img_width'], # TODO: all the img is the same size
                                                                    'img_height': meta_data['img_height'],
                                                                    'net_width': self.params_transform['crop_size_x'],
                                                                    'net_height': self.params_transform['crop_size_y']}  # 8 terms


        

        #     # output the multiple views



        # # generate labels
        # valid = np.ones((self.keypoint_num +self.paf_num*3, 1), np.float)  
        # if meta_data['dataset'] == 'COCO':
        #     # coco has no headtop annotation
        #     valid[1, 0] = 0  
        #     # pafs of headtop and neck
        #     valid[self.keypoint_num, 0] = 0   
        #     valid[self.keypoint_num+1, 0] = 0
        #     # relative depth
        #     valid[self.keypoint_num + self.paf_num*2:, 0] = 0

        # labels_num = len(self.gaussian_kernels) # labels_num is the person's number?
        # labels = np.zeros((labels_num, self.keypoint_num + self.paf_num*3, *self.output_shape))  # TODO: The use of different guassian kernel
        # for i in range(labels_num): # 不同 kernel size 下
        #     # heatmaps
        #     labels[i][:self.keypoint_num] = generate_heatmap(meta_data['bodys'], self.output_shape, self.stride, \
        #                                                      self.keypoint_num, kernel=self.gaussian_kernels[i])  # heatmap channel
        #     # pafs + relative depth
        #     labels[i][self.keypoint_num:] = generate_paf(meta_data['bodys'], self.output_shape, self.params_transform, \
        #                                                  self.paf_num, self.paf_vector, max(1, (3-i))*self.paf_thre, self.with_mds)
        
        # # root depth
        # labels_rdepth = generate_rdepth(meta_data, self.stride, self.root_idx, self.max_people)
        
        # labels = torch.from_numpy(labels).float()
        # labels_rdepth = torch.from_numpy(labels_rdepth).float()
        # valid = torch.from_numpy(valid).float()

        # return img, valid, labels, labels_rdepth








