import torch
import pickle
import os
from tqdm import tqdm
import numpy as np

path_dir = '/Extra/panzhiyu/CMU_data/gnn_test'

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


# path_dir = '/Extra/panzhiyu/Shelf/shelf_result.pkl'
# with open(path_dir, 'rb') as f:
#     result = pickle.load(f)
#     result = result['3d_pairs']

# new_result = dict()
# for v in range(5):
#     new_result[v] = dict()

# for per_result in tqdm(result):
#     init_image = per_result['img_path']
#     view_r = init_image.split('/')[-2]
#     view_k = int(view_r.replace('Camera',''))
#     # image_num = init_image.split('/')[-1].replace(view_r+'_','')
#     image_num = int(init_image.split('/')[-1].split('_')[-1].split('.png')[0])
#     new_result[view_k][image_num] = per_result


# new_file = '/Extra/panzhiyu/Shelf/shelf_result_new.pkl'
# with open(new_file,'wb') as f:
#     pickle.dump(new_result, f)
    