import json
import os.path as osp
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import os
from tqdm import tqdm

pairs = [[0, 1], [0, 2], [0, 9], [9, 10], [10, 11],
         [0, 3], [3, 4], [4, 5], [2, 12], [12, 13], 
         [13, 14], [2, 6], [6, 7], [7, 8]]

colors = ['r', 'g', 'b', 'y', 'k', 'p']

json_file = '/home/panzhiyu/project/3d_pose/SMAP/model_logs_campusE2/stage3_root2/validation_result/campus_aligned.json'


with open(json_file,'r') as f:
    data = json.load(f)['3d_pairs']

num_views = 3
output_folder = '/Extra/panzhiyu/CampusSeq1/demo_folderv2'
for idata in tqdm(data):
    # pred_3d = np.array(idata['pred_3d'])
    pred_3d = np.array(idata['pred_aligned_g3d'])
    gt_3d = np.array(idata['gt_g3d'])
    img_path = osp.join(idata['image_path'])
    # only plot view 0
    # get the folder name
    file_name = os.path.split(img_path)[1]
    view_lab = file_name[9]
    # import pdb; pdb.set_trace()
    if int(view_lab) != 2:
        continue 
    output_file = os.path.join(output_folder, file_name)
    
    img = cv2.imread(img_path)[:, :, ::-1]

    fig = plt.figure(figsize=(20, 15)) # 20 15
    ax1 = fig.add_subplot(131)
    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(132, projection='3d')
    for ip in range(len(pred_3d)):
        p3d = pred_3d[ip]
        for pair in pairs:
            if p3d[pair[0],3] == 0 or p3d[pair[1],3] == 0:
                continue
            ax2.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c='r')  #[ip%len(colors)]
    # ax2.view_init(elev=-45) #,azim=-90 
    ax2.set_xlim([-3,5])
    ax2.set_ylim([1,10]) 
    ax2.set_zlim([-0.5,2])
    ax2.set_title('prediction', fontsize = 20) #fontproperties = "bold"

    ax3 = fig.add_subplot(133, projection='3d')
    for ip in range(len(gt_3d)):
        p3d = gt_3d[ip]
        for pair in pairs:
            ax3.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c='b')  #colors[ip%len(colors)]
    # ax3.view_init(azim=-90, elev=-45)
    ax3.set_xlim([-3,5])
    ax3.set_ylim([1,10]) 
    ax3.set_zlim([-0.5,2])
    ax3.set_title('ground truth', fontsize = 20) #fontproperties ="bold",
    plt.savefig(output_file)
    plt.close() 
    

