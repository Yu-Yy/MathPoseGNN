import torch
from torch.utils.data import Dataset
import os
import pickle

class GNNdataset(Dataset): # train and test
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.file_list = os.listdir(self.data_dir)
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        single_file = self.file_list[index]
        with open(os.path.join(self.data_dir, single_file),'rb') as f:
            gnn_pair = pickle.load(f)
        matched_pred_single = gnn_pair['pred_single']
        matched_pred3d = gnn_pair['pred_3d']
        gt_3d = gnn_pair['gt_3d']
        gt_bodys_2d = gnn_pair['gt_2d']
        cam_info = gnn_pair['cam']

        return matched_pred_single, matched_pred3d, gt_3d, gt_bodys_2d, cam_info
        

        
