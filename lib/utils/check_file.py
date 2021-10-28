import pickle
import torch
import numpy as np
import os

file1 = '/Extra/panzhiyu/CMU_data/gnn_test/gnn_train_00000.pkl'
# files = os.listdir(dir)
with open(os.path.join(file1),'rb') as f:
    result1 = pickle.load(f)

file2 = '/Extra/panzhiyu/CMU_data/gnn_hmtest/gnn_train_00000.pkl'
# files = os.listdir(dir)
with open(os.path.join(file2),'rb') as f:
    result2 = pickle.load(f)    

file3 = '/home/panzhiyu/project/3d_pose/SMAP/model_logs_0821/stage3_root2/validation_result/stage3_root2_generate_result_test_orig.pkl'
with open(file3,'rb') as f:
    result3 = pickle.load(f)   


print('test')