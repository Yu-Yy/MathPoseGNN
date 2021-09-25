"""
@author: Jianan Zhen
@contact: zhenjianan@sensetime.com
"""

import os
import argparse
import json
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader

from cvpack.utils.logger import get_logger
from model.smap import SMAP
from model.refinenet import RefineNet
from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process
from exps.stage3_root2.test_util import *
from dataset.custom_dataset import CustomDataset
from config import cfg
# import datalib
import dapalib
import cv2
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits import mplot3d
import pickle
import torch.nn.functional as F
from scipy.spatial import procrustes


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

colors = ['r', 'g', 'b', 'y', 'k', 'p']

def generate_3d_point_pairs(model, refine_model, data_loader, cfg, logger, device, 
                            output_dir='', total_iter='infer'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    if refine_model is not None:
        refine_model.eval()

    result = dict()
    result['model_pattern'] = cfg.DATASET.NAME
    result['3d_pairs'] = []
    # 3d_pairs has items like{'pred_2d':[[x,y,detZ,score]...], 'gt_2d':[[x,y,Z,visual_type]...],
    #                         'pred_3d':[[X,Y,Z,score]...], 'gt_3d':[[X,Y,X]...],
    #                         'root_d': (abs depth of root (float value) pred by network),
    #                         'image_path': relative image path}

    kpt_num = cfg.DATASET.KEYPOINT.NUM
    data = tqdm(data_loader) if is_main_process() else data_loader
    # create the val dict
    kp_set = [AverageMeter() for _ in range(cfg.DATASET.KEYPOINT.NUM)]
    mpjpe = AverageMeter()
    for idx, batch in enumerate(data):
        # eval the mpjpe
        # if idx <= 5000:
        #     continue
        # if cfg.TEST_MODE == 'run_inference':
        #     imgs, img_path, scales = batch
        #     meta_data = None
        # else:
        # imgs, meta_data, img_path, scales = batch #
        
        imgs_mul, cam_paras, joints3d_global, joints3d_local_mul, scales = batch
        imgs_mul = imgs_mul.to(device)
        batch_size = imgs_mul.shape[0]
        with torch.no_grad():
            # in multiple views
            for idx_v ,node in enumerate(cfg.DATASET.CAM):
                imgs = imgs_mul[:,idx_v,...]
                meta_data = joints3d_local_mul[:,idx_v,...]
                cam_p = cam_paras[0][idx_v] # the parameters are same
                Kd = cam_p['distCoef']
                outputs_2d, outputs_3d, outputs_rd, outputs_feature = model(imgs)

                outputs_3d = outputs_3d.cpu()
                outputs_rd = outputs_rd.cpu()

                # if cfg.DO_FLIP:
                #     imgs_flip = torch.flip(imgs, [-1])
                #     outputs_2d_flip, outputs_3d_flip, outputs_rd_flip = model(imgs_flip)
                #     outputs_2d_flip = torch.flip(outputs_2d_flip, dims=[-1])
                #     # outputs_3d_flip = torch.flip(outputs_3d_flip, dims=[-1])
                #     # outputs_rd_flip = torch.flip(outputs_rd_flip, dims=[-1])
                #     keypoint_pair = cfg.DATASET.KEYPOINT.FLIP_ORDER
                #     paf_pair = cfg.DATASET.PAF.FLIP_CHANNEL
                #     paf_abs_pair = [x+kpt_num for x in paf_pair]
                #     pair = keypoint_pair + paf_abs_pair
                #     for i in range(len(pair)):
                #         if i >= kpt_num and (i - kpt_num) % 2 == 0:
                #             outputs_2d[:, i] += outputs_2d_flip[:, pair[i]]*-1
                #         else:
                #             outputs_2d[:, i] += outputs_2d_flip[:, pair[i]]
                #     outputs_2d[:, kpt_num:] *= 0.5

                for i in range(len(imgs)):
                    if meta_data is not None: # meta data is the joints3d_local
                        # remove person who was blocked
                        new_gt_bodys = []
                        annotation = meta_data[i].numpy()
                        scale = scales[i]
                        for j in range(len(annotation)):
                            if annotation[j, cfg.DATASET.ROOT_IDX, 3] > 1:
                                new_gt_bodys.append(annotation[j])
                        gt_bodys = np.asarray(new_gt_bodys)
                        if len(gt_bodys) == 0:
                            continue
                        # groundtruth:[person..[keypoints..[x, y, Z, score(0:None, 1:invisible, 2:visible), X, Y, Z,
                        #                                   f_x, f_y, cx, cy]]]
                        if len(gt_bodys[0][0]) < 11:
                            scale['f_x'] = gt_bodys[0, 0, 7]
                            scale['f_y'] = gt_bodys[0, 0, 7]
                            scale['cx'] = scale['img_width']/2
                            scale['cy'] = scale['img_height']/2
                        else:
                            scale['f_x'] = gt_bodys[0, 0, 7]
                            scale['f_y'] = gt_bodys[0, 0, 8]
                            scale['cx'] = gt_bodys[0, 0, 9]
                            scale['cy'] = gt_bodys[0, 0, 10]
                    else:
                        gt_bodys = None
                        # use default values
                        scale = {k: scales[k][i].numpy() for k in scales}
                        scale['f_x'] = scale['img_width']
                        scale['f_y'] = scale['img_width']
                        scale['cx'] = scale['img_width']/2
                        scale['cy'] = scale['img_height']/2


                    hmsIn = outputs_2d[i]
                    # if the first pair is [1, 0], uncomment the code below
                    # hmsIn[cfg.DATASET.KEYPOINT.NUM:cfg.DATASET.KEYPOINT.NUM+2] *= -1
                    # outputs_3d[i, 0]n *= -1
                    hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255
                    hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127
                    rDepth = outputs_rd[i][0]
                    # no batch implementation yet  # depth aware?
                    pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True) # depth-aware part association lib # 存在预测分数
                    
                    if len(pred_bodys_2d) > 0:
                        pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape 4
                        pred_bodys_2d = pred_bodys_2d.numpy()

                    pafs_3d = outputs_3d[i].numpy().transpose(1, 2, 0)
                    root_d = outputs_rd[i][0].numpy() # changed cm to m

                    paf_3d_upsamp = cv2.resize(
                        pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST) # get the rela depth map
                    root_d_upsamp = cv2.resize(
                        root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)

                    # generate 3d prediction bodys
                    pred_bodys_2d = register_pred(pred_bodys_2d, None) #gt_bodys # make all the pred be zero
                    
                    if len(pred_bodys_2d) == 0:
                        continue
                    pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scale, idx_v) # pred_bodys_2d with relative depth
                    pred_bodys_3d, adjust_2d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scale, idx_v, Kd)

                    if refine_model is not None:
                        new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                                                                    device=device, root_n=cfg.DATASET.ROOT_IDX)
                    else:
                        new_pred_bodys_3d = pred_bodys_3d

                    # if cfg.TEST_MODE == "generate_train":
                    #     save_result_for_train_refine(pred_bodys_2d, new_pred_bodys_3d, gt_bodys, pred_rdepths, result)
                    # else:
                        # vis the data
                    # gt_3d amd new_pred_bodys_3d (分关节 和 总共)
                    val(torch.Tensor(pred_bodys_3d), torch.Tensor(gt_bodys[:,:,3:7]), mpjpe, kp_set)
                    # import pdb; pdb.set_trace()
                    # set the global coordinate to test the project , 
                    
                    # global_pose3d = np.ones(pred_bodys_3d.shape)
                    # global_pose3d[:,:,3] = pred_bodys_3d[:,:,3]
                    # global_pose3d[:,:,:3] = (np.linalg.pinv(cam_p['R']) @ (pred_bodys_3d[:,:,:3].reshape(-1,3).transpose() - \
                    #     cam_p['t'])).transpose().reshape(-1,15,3)
                    # # using procrute alignement to get the final result
                    gt_3d = gt_bodys[:,:,4:7]
                    # gt_global3d = (np.linalg.pinv(cam_p['R']) @ (gt_3d[:,:,:3].reshape(-1,3).transpose() - \
                    #     cam_p['t'])).transpose().reshape(-1,15,3)
                    # # pro align not consider the nonvalid point
                    # aligned_pose = PA(gt_global3d, global_pose3d)


                    if idx % 500 == 0:
                        joint_folder = os.path.join(output_dir,'joints_folder')
                        os.makedirs(joint_folder, exist_ok=True)
                        vis_joints3d(imgs[i], new_pred_bodys_3d, gt_3d, joint_folder, idx, i, idx_v, total_iter = total_iter)
                        # vis_joints3d(imgs[i], aligned_pose, gt_global3d, joint_folder, idx, i, idx_v, total_iter = total_iter)
                    import pdb; pdb.set_trace()
                    # img_paths = scale['img_paths'][idx_v]
                    # save_result(pred_bodys_2d, new_pred_bodys_3d, global_pose3d, aligned_pose, gt_bodys, gt_global3d, pred_rdepths, img_paths,result) #img_path[i],
                
                if idx % 500 == 0:
                    vis_2d(hmsIn, imgs[i], root_d_upsamp,output_dir,idx, idx_v, total_iter = total_iter)



    # save the final result in json file
    # dir_name = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]
    # pair_file_name = os.path.join(output_dir, '{}_{}_{}_{}.json'.format(dir_name, cfg.TEST_MODE,
    #                                                                     cfg.DATA_MODE, cfg.JSON_SUFFIX_NAME))
    # with open(pair_file_name, 'w') as f:
    #     json.dump(result, f)
    # logger.info("Pairs writed to {}".format(pair_file_name))

    
    msg =  f'EPOCH {total_iter}: The MPJPE is {mpjpe.avg}'
    logger.info(msg)
    for i in range(kpt_num):
        logger.info(f'The {key_name[i]} is {kp_set[i].avg}')

    return mpjpe.avg

def vis_joints3d(img_t, pred_3d, gt_3d, output_dir, idx, count, view_idx ,total_iter=0):# img_path,
    # img = cv2.imread(img_path)[:, :, ::-1]
    img = tensor2im(img_t)
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(141)
    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(142, projection='3d')
    for ip in range(len(pred_3d)):
        p3d = pred_3d[ip]
        for pair in pairs:
            # 存在
            ax2.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
    ax2.view_init(azim=-90, elev=-45)

    ax3 = fig.add_subplot(143, projection='3d')
    for ip in range(len(gt_3d)):
        p3d = gt_3d[ip]
        for pair in pairs:
            ax3.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)]) # linestyle=':'
    ax3.view_init(azim=-90, elev=-45)

    ax4 = fig.add_subplot(144, projection='3d')
    for ip in range(len(pred_3d)):
        p3d = pred_3d[ip]
        for pair in pairs:
            ax4.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)], linestyle= ':') # linestyle=':'
    for ip in range(len(gt_3d)):
        p3d = gt_3d[ip]
        for pair in pairs:
            ax4.plot(p3d[pair, 0], p3d[pair, 1], p3d[pair, 2], c=colors[ip%len(colors)])
    ax4.view_init(azim=-90, elev=-45)

    
    try:
        plt.savefig(os.path.join(output_dir,f'pose3d_iter_{total_iter}_i_{idx}_c_{count}_view_{view_idx}.jpg'))
    except:
        # print(img.shape)
        pass
    plt.close(fig)

def val(pred_3d, gt_3d, mpjpe, kp_set):
    pred_num = pred_3d.shape[0]
    pred_3d_vislabel = pred_3d[:,:,3]
    pred_3d = pred_3d[:,:,:3]
    # import pdb;pdb.set_trace()
    pred_3d_cn = pred_3d.reshape(pred_num,-1)
    pred_3d_cn = pred_3d_cn[:,None,:]
    actual_num = gt_3d.shape[0]
    pred_3d_cn = pred_3d_cn.repeat(1,actual_num,1)
    gt_3d_position = gt_3d[:,:,1:]
    gt_3d_cn = gt_3d_position.reshape(actual_num,-1) # long vector
    gt_valid = gt_3d[:,:,0]
    dist = torch.norm((pred_3d_cn - gt_3d_cn),dim=-1)
    match_idx = torch.argsort(dist, dim=-1)[:,0]
    for i in range(pred_num):
        match_i = match_idx[i] # 
        for j in range(cfg.DATASET.KEYPOINT.NUM):
            if gt_valid[match_i, j] !=0 and pred_3d_vislabel[i,j]!=0:
                error = torch.norm((gt_3d_position[match_i,j,:] - pred_3d[i,j,:])) # invisible joint process?
                kp_set[j].update(error.item())
                mpjpe.update(error.item())

def vis_2d(hmsIn, img, root_d, folder_name, idx, idx_v, total_iter =0):
    # visualization
    # folder_name = os.path.join(cfg.OUTPUT_DIR, 'debug_test_pics')
    hm_folder = os.path.join(folder_name,'heatmap')
    paf_folder = os.path.join(folder_name,'paf') # contains the
    rootd_folder = os.path.join(folder_name,'rootdepth')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(hm_folder):
        os.makedirs(hm_folder)
    if not os.path.exists(paf_folder):
        os.makedirs(paf_folder)
    if not os.path.exists(rootd_folder):
        os.makedirs(rootd_folder)

    hmsIn = hmsIn.detach().cpu().numpy().transpose(1, 2, 0)
    hmsIn_upsamp = cv2.resize(
                    hmsIn, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
    vis_pic = tensor2im(img)
    # get the heatmap, paf, and relative depth map
    heatmap_vis = hmsIn_upsamp[...,:cfg.DATASET.KEYPOINT.NUM]
    paf_vis = hmsIn_upsamp[...,cfg.DATASET.KEYPOINT.NUM:]
    # rdepth_vis = outputs['det_d'][-1][-1][0]

    # root depth did not exist the vis map, just show the final joint 
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(vis_pic)
    plt.subplot(122)
    plt.imshow(root_d)
    plt.savefig(os.path.join(rootd_folder, f'rdepth_{total_iter}_i_{idx}_v_{idx_v}.jpg'))
    plt.close(fig)

    for hm_idx in range(cfg.DATASET.KEYPOINT.NUM):
        vis_hm = torch.tensor(heatmap_vis[...,hm_idx]).mul(255).clamp(0, 255).byte().numpy()
        colored_heatmap = cv2.applyColorMap(vis_hm, cv2.COLORMAP_JET)
        image_fusing = vis_pic * 0.3 + colored_heatmap * 0.7

        cv2.imwrite(os.path.join(hm_folder, f'hm_iter_{total_iter}_i_{idx}_v_{idx_v}_joints_{key_name[hm_idx]}.jpg'), image_fusing)

    for con_idx in range(cfg.DATASET.PAF.NUM):
        # background = np.zeros((h,w))
        vis_paf = np.sum(np.abs(paf_vis[...,con_idx*2:con_idx*2+2]*127 ),axis=-1)
        vis_paf = torch.tensor(vis_paf).clamp(0, 255).byte().numpy()
        # .detach().cpu().numpy()

        colored_bool = cv2.applyColorMap(vis_paf, cv2.COLORMAP_JET)
        image_fusing = vis_pic * 0.3 + colored_bool * 0.7
        cv2.imwrite(os.path.join(paf_folder, f'paf_iter_{total_iter}_i_{idx}_v_{idx_v}_paf_{con_idx}.jpg'), image_fusing)    

def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = cfg.INPUT.MEANS #dataLoader中设置的mean参数
    std = cfg.INPUT.STDS  #dataLoader中设置的std参数
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): #如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.detach().cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): #反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
        temp = image_numpy[0,...].copy()
        image_numpy[0,...] = image_numpy[2,...].copy()
        image_numpy[2,...] = temp
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)

def PA(gt_3d, global_pose3d):
    # match the global_pose3d to the gt_global3d
    pred_num = global_pose3d.shape[0]
    # pred_3d_vislabel = global_pose3d[:,:,3]
    pred_3d = global_pose3d[:,:,:3]
    # import pdb;pdb.set_trace()
    pred_3d_cn = torch.tensor(pred_3d.reshape(pred_num,-1))
    pred_3d_cn = pred_3d_cn[:,None,:]
    actual_num = gt_3d.shape[0]
    pred_3d_cn = pred_3d_cn.repeat(1,actual_num,1)
    gt_3d_position = gt_3d #[:,:,1:]
    gt_3d_cn = torch.tensor(gt_3d_position.reshape(actual_num,-1)) # long vector
    # gt_valid = gt_3d[:,:,0]
    dist = torch.norm((pred_3d_cn - gt_3d_cn),dim=-1)
    match_idx = torch.argsort(dist, dim=-1)[:,0]
    matched_pose = np.ones(global_pose3d.shape)
    for i in range(pred_num):
        match_i = match_idx[i] #
        matched_pose[i,:,3] = global_pose3d[i,:,3]
        extracted_pred = global_pose3d[i,:,:3]
        extracted_gt = gt_3d[match_i,...]
        valid_label = (global_pose3d[i,:,3]!=0)
        if np.sum(valid_label) <= 3:
            matched_pose[i,:,:3] = extracted_pred
        else:
            mtx1, mtx2, disparity, matched_data = procrustes(extracted_gt[global_pose3d[i,:,3]!=0], extracted_pred[global_pose3d[i,:,3]!=0])
            matched_pose[i,:,:3][global_pose3d[i,:,3]!=0] = matched_data

        # for j in range(cfg.DATASET.KEYPOINT.NUM):
        #     if gt_valid[match_i, j] !=0 and pred_3d_vislabel[i,j]!=0:
        #         error = torch.norm((gt_3d_position[match_i,j,:] - pred_3d[i,j,:])) # invisible joint process?
        #         kp_set[j].update(error.item())
        #         mpjpe.update(error.item())
    return matched_pose

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", "-t", type=str, default="run_inference",
                        choices=['generate_train', 'generate_result', 'run_inference'],
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

    model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if args.test_mode == "run_inference":
        test_dataset = CustomDataset(cfg, args.dataset_path)
        data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # print('use dataset!')
        data_loader = get_test_loader(cfg, num_gpu=1, local_rank=0, stage=args.data_mode)

    if cfg.REFINE:
        refine_model = RefineNet()
        refine_model.to(device)
        refine_model_file = args.RefineNet_path
    else:
        refine_model = None
        refine_model_file = ""

    model_file = args.SMAP_path
    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        if os.path.exists(refine_model_file):
            refine_model.load_state_dict(torch.load(refine_model_file))
        elif refine_model is not None:
            logger.info("No such RefineNet checkpoint of {}".format(args.RefineNet_path))
            return
        generate_3d_point_pairs(model, refine_model, data_loader, cfg, logger, device,
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

if __name__ == '__main__':
    main()
