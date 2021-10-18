import torch
import pickle
import os
from tqdm import tqdm

path_dir = '/Extra/panzhiyu/CMU_data/gnn_train4'

list_file = os.listdir(path_dir)
for file in tqdm(list_file):
    with open(os.path.join(path_dir, file), 'rb') as f:
        gnn_pair = pickle.load(f)
    # gt_2d = 
    # cam_info = 
    new_cam_info = dict()
    new_gt_2d = dict()
    for k in gnn_pair['gt_2d'].keys():
        gnn_pair['gt_2d'][k] = gnn_pair['gt_2d'][k].cpu()
        for n_k in gnn_pair['cam'][k].keys():
            gnn_pair['cam'][k][n_k] = gnn_pair['cam'][k][n_k].cpu()
    # reload
    with open(os.path.join(path_dir, file), 'wb') as f:
        pickle.dump(gnn_pair,f)
    # update the gt_2d ,cam_info