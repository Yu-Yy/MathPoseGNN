import argparse
import time
import numpy as np
import cv2
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from cvpack.torch_modeling.engine.engine import Engine
from cvpack.utils.pyt_utils import ensure_dir

from config import cfg
from model.smap import SMAP
from model.smap_gnn import SMAP_GNN
from lib.utils.dataloader import get_train_loader, get_test_loader
from lib.utils.solver import make_lr_scheduler, make_optimizer
from test_ import generate_3d_point_pairs
from cvpack.utils.logger import get_logger
# for vist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

from dataset.custom_dataset import CustomDataset
# debug the file error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib.utils.comm import is_main_process
from tqdm import tqdm
kp_align_num = 2048

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
    'r-ankle': 14}
key_name = list(JOINTS_DEF.keys())
pairs = [[0, 1], [0, 2], [0, 9], [9, 10], [10, 11],
         [0, 3], [3, 4], [4, 5], [2, 12], [12, 13], 
         [13, 14], [2, 6], [6, 7], [7, 8]]

colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']

# === common joints definition between panoptic and coco
panoptic_to_unified = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
coco_to_unified = [0, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]
unified_joints_def = {
    'nose': 0,
    'l-shoulder': 1,
    'l-elbow': 2,
    'l-wrist': 3,
    'l-hip': 4,
    'l-knee': 5,
    'l-ankle': 6,
    'r-shoulder': 7,
    'r-elbow': 8,
    'r-wrist': 9,
    'r-hip': 10,
    'r-knee': 11,
    'r-ankle': 12,
}
unified_bones_def = [
    [0, 1], [0, 7],  # head
    [1, 2], [2, 3],  # left arm
    [7, 8], [8, 9],  # right arm
    [1, 4], [7, 10],  # trunk
    [4, 5], [5, 6],  # left leg
    [10, 11], [11, 12],  # right leg
]

unified_bones_def14 = [
    [0, 1],
    [0, 2], [0, 8],  # head
    [2, 3], [3, 4],  # left arm
    [8, 9], [9, 10],  # right arm
    [2, 5], [8, 11],  # trunk
    [5, 6], [6, 7],  # left leg
    [11, 12], [12, 13],  # right leg
]

File_Name = '/Extra/panzhiyu/CMU_data/keypoints_validation_results.json'

OFF_FILE = '/home/panzhiyu/project/3d_pose/SMAP/keypoints_validation_results.pkl'

with open(OFF_FILE,'rb') as f:
    pred_2d_results = pickle.load(f)


def generate_3d_point_pairs(model, refine_model, data_loader, cfg, logger, device, 
                            output_dir='', total_iter='infer'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    if refine_model is not None:
        refine_model.eval()

    data = tqdm(data_loader) if is_main_process() else data_loader
    root_dir = '/Extra/panzhiyu/CMU_data/'
    mpjpe = AverageMeter()
    with torch.no_grad():
        for idx, batch in enumerate(data):
            imgs_mul, cam_paras, joints3d_global, joints3d_local_mul, scales = batch
            imgs_mul = imgs_mul.to(device)

            c_image = scales[0]['img_paths'][0].split(root_dir)[-1]

            if c_image not in pred_2d_results.keys():
                continue

            joints3d_global= joints3d_global.to(device)
            batch_size = imgs_mul.shape[0]
            _, refined_pose3d, matched_gt= model(imgs_mul, None, None, None, cam_paras, joints3d_global, joints3d_local_mul, scales)

            pred_count = torch.sum(refined_pose3d[:,:,0,3],dim=-1, keepdim=True )
            valid_label = (refined_pose3d[:,:,:,3:4] > 0) * (matched_gt[:,:,:,3:4] > 0.1)
            loss_temp = torch.norm((refined_pose3d[:,:,:,:3] - matched_gt[:,:,:,:3]) * valid_label, dim=-1)
            loss_m1 = torch.mean(loss_temp, dim=-1)
            mp_500 = torch.sum((loss_m1 > 50), dim=-1, keepdim=True)   
            pred_count = pred_count - mp_500
            mpjpe_C = torch.mean(torch.sum(loss_m1 * (1/pred_count), dim = -1)) # 0.01 is the weight
            import pdb; pdb.set_trace()
            if torch.isnan(mpjpe_C):
                continue
            mpjpe.update(mpjpe_C.item())
    print(f'The mpjpe is {mpjpe.avg}')
    print(f'The mpjpe is {mpjpe.avg}')
    print(f'The mpjpe is {mpjpe.avg}')
    print(f'The mpjpe is {mpjpe.avg}')
    print(f'The mpjpe is {mpjpe.avg}')
    print(f'The mpjpe is {mpjpe.avg}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", "-t", type=str, default="run_inference",
                        choices=['generate_train', 'generate_result', 'run_inference', 'off_the_shelf'],
                        help='Type of test. One of "generate_train": generate refineNet datasets, '
                             '"generate_result": save inference result and groundtruth, '
                             '"run_inference": save inference result for input images.')
    parser.add_argument("--data_mode", "-d", type=str, default="test",
                        choices=['test', 'generation'],
                        help='Only used for "generate_train" test_mode, "generation" for refineNet train dataset,'
                             '"test" for refineNet test dataset.')
    parser.add_argument("--SMAP_path", "-p", type=str, default='log/SMAP.pth',
                        help='Path to SMAP model')
    parser.add_argument("--RefineNet_path", "-rp", type=str, default='',
                        help='Path to RefineNet model, empty means without RefineNet')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch_size of test')
    parser.add_argument("--do_flip", type=float, default=0,
                        help='Set to 1 if do flip when test')
    parser.add_argument("--dataset_path", type=str, default="",
                        help='Image dir path of "run_inference" test mode')
    parser.add_argument("--json_name", type=str, default="",
                        help='Add a suffix to the result json.')
    args = parser.parse_args()
    cfg.TEST_MODE = args.test_mode
    cfg.DATA_MODE = args.data_mode
    cfg.REFINE = len(args.RefineNet_path) > 0
    cfg.DO_FLIP = args.do_flip
    cfg.JSON_SUFFIX_NAME = args.json_name
    cfg.TEST.IMG_PER_GPU = args.batch_size

    os.makedirs(cfg.TEST_DIR, exist_ok=True)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, 0, 'test_log_{}.txt'.format(args.test_mode))

    device = torch.device(cfg.MODEL.DEVICE)
    if args.test_mode == "off_the_shelf":
        pose_file = os.path.join('/home/panzhiyu/project/3d_pose/SMAP/model_logs_0821/stage3_root2/validation_result/','stage3_root2_generate_result_test_orig.pkl')
        # easy_mode(pose_file, cfg, logger, device, output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result"))
    else:
        model = SMAP_GNN(shift_size=3, is_train=False)

        model.to(device)

        if args.test_mode == "run_inference":
            test_dataset = CustomDataset(cfg, args.dataset_path)
            data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            # print('use dataset!')
            data_loader = get_test_loader(cfg, num_gpu=1, local_rank=0, stage=args.data_mode)


        model_file = args.SMAP_path
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = state_dict['model']
            model.load_state_dict(state_dict)

            generate_3d_point_pairs(model, None, data_loader, cfg, logger, device,
                                    output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result")) #os.path.join(cfg.OUTPUT_DIR, "result")
        else:
            logger.info("No such checkpoint of SMAP {}".format(args.SMAP_path))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FPAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()