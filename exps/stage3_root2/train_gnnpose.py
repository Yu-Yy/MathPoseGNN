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
from model.pose2d_gnn import Pose_GCN
from lib.utils.dataloader import get_train_loader, get_test_loader, get_traingnn_loader
from lib.utils.solver import make_lr_scheduler, make_optimizer
from test_ import generate_3d_point_pairs
from test_util import project_views
from pc_gen import easy_mode

# for vist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

def main():
    parser = argparse.ArgumentParser()
    # torch.multiprocessing.set_start_method('spawn')
    with Engine(cfg, custom_parser=parser) as engine:
        logger = engine.setup_log(
            name='train', log_dir=cfg.OUTPUT_DIR, file_name='log.txt')
        args = engine.args
        ensure_dir(cfg.OUTPUT_DIR)

        # model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
        model = Pose_GCN(inplaneS=7, inplaneM=4, device=torch.device('cuda')) # network input 
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)

        num_gpu = len(engine.devices)
        # import pdb;pdb.set_trace()
        #  default num_gpu: 8, adjust iter settings
        cfg.SOLVER.CHECKPOINT_PERIOD = \
                int(cfg.SOLVER.CHECKPOINT_PERIOD * 2/ num_gpu) # TODO: 8 -> 1 # divide the view num

        # cfg.SOLVER.CHECKPOINT_PERIOD = 10000
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * 2/ num_gpu)

        logger.info(f'The checkpoint period is {cfg.SOLVER.CHECKPOINT_PERIOD }')
        logger.info(f'max_iter period is {cfg.SOLVER.MAX_ITER}')

        optimizer = make_optimizer(cfg, model, num_gpu)
        scheduler = make_lr_scheduler(cfg, optimizer)

        # optimizer2 = make_optimizer(cfg, model.gnn_net, num_gpu)
        # scheduler2 = make_lr_scheduler(cfg, optimizer2)

        engine.register_state( # provide the SMAP info
            scheduler=scheduler, model=model, optimizer=optimizer)

        if engine.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                broadcast_buffers=False)

        # load the pretrained model
        if cfg.Pretrained:
            model_file = '/home/panzhiyu/project/3d_pose/SMAP/model_logs_1016New/stage3_root2/best_model.pth'
            state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = state_dict['model']
            model.module.load_state_dict(state_dict)
            logger.info(f'Loading the pretrained model from {model_file}')
            
        if engine.continue_state_object: 
            engine.restore_checkpoint(is_restore=False)
        # else:
        #     if cfg.MODEL.WEIGHT: # load the SMAP pretrained model
        #         # engine.load_checkpoint(cfg.MODEL.WEIGHT, is_restore=False)
        #         # logger.info(f'load the pretrained model from {cfg.MODEL.WEIGHT}') 
        #         model.preload(cfg.MODEL.WEIGHT) # just load the SMAP part .module.module.

        data_dir = '/Extra/panzhiyu/CMU_data/gnn_train3'
        train_data_loader = get_traingnn_loader(cfg, num_gpu=num_gpu, data_dir=data_dir, is_dist=engine.distributed) # 图片增强， MDS true
        # vali_data_loader = get_test_loader(cfg, num_gpu=num_gpu, local_rank=0, stage='test') # TODO : temp comment

        # print('load loader ok')
        # -------------------- do training -------------------- #
        logger.info("\n\nStart training with pytorch version {}".format(
            torch.__version__))

        max_iter = len(train_data_loader) # the length of data_loader
        logger.info(f'The length of the loader is {max_iter}')
        # max_iter = max_iter / 4 # for testing 

        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        if engine.local_rank == 0:
            tb_writer = SummaryWriter(cfg.TENSORBOARD_DIR)
        model.train()
        time1 = time.time()
        # the concept of batch?
        # import pdb;pdb.set_trace() 不设置epoch概念
        # epoch = 1
        best_mpjpe = 10000

        for iteration, (matched_pred_single, matched_pred3d, gt_3d, gt_bodys_2d, cam_info) in enumerate(
                train_data_loader, engine.state.iteration):
            iteration = iteration + 1
            for v in matched_pred_single.keys():
                matched_pred_single[v] = matched_pred_single[v].reshape(-1,cfg.DATASET.MAX_PEOPLE,cfg.DATASET.KEYPOINT.NUM,7)
                matched_pred_single[v] = matched_pred_single[v].to(device)
                gt_bodys_2d[v] = gt_bodys_2d[v].to(device)
                gt_bodys_2d[v] = gt_bodys_2d[v].reshape(-1,cfg.DATASET.MAX_PEOPLE,cfg.DATASET.KEYPOINT.NUM,3)
                for attr in cam_info[v].keys():
                    cam_info[v][attr] = cam_info[v][attr][0,...].to(device) # just one batch
            
            matched_pred3d = matched_pred3d.to(device)
            matched_pred3d = matched_pred3d.reshape(-1,cfg.DATASET.MAX_PEOPLE, cfg.DATASET.KEYPOINT.NUM,4)
            gt_3d = gt_3d.to(device)
            gt_3d = gt_3d.reshape(-1,cfg.DATASET.MAX_PEOPLE, cfg.DATASET.KEYPOINT.NUM,4)
            final_pred3d = model(matched_pred_single, matched_pred3d)
            final_pred3d = torch.cat([final_pred3d, matched_pred3d[...,3:4]], dim=-1)
            # project and get the pred2d
            final_pred2d = project_views(final_pred3d, cam_info)
            
            # # loss_3d part 
            gt_3d_vis = (gt_3d[...,3] > 0) 
            temp_loss_3d =  torch.norm(final_pred3d[...,:3] - gt_3d[...,:3], dim=-1) ** 2 * gt_3d_vis
            valid_people = torch.sum(temp_loss_3d > 0 )
            loss_3d = torch.sum(temp_loss_3d) / valid_people

            temp_d_collect = []
            for k in final_pred2d.keys():
                # final_pred2d_vis = (final_pred2d[k][...,2] > 0)
                gt_2d_vis = (gt_bodys_2d[k][...,2] > 0.1)
                temp_loss_2d = torch.norm(final_pred2d[k][...,:2] - gt_bodys_2d[k][...,:2], dim=-1) ** 2 * gt_2d_vis
                temp_d_collect.append(temp_loss_2d)
                # valid_p = torch.sum(temp_loss_2d > 0)
                # loss_2d = loss_2d + torch.sum(temp_loss_2d) / valid_p
            temp_d_collect = torch.cat(temp_d_collect, dim=0)
            valid_p = torch.sum(temp_d_collect > 0)
            loss_2d = torch.sum(temp_d_collect) / valid_p

            loss_2d = loss_2d / len(final_pred2d.keys())
            losses =  loss_3d  #0.01 * loss_2d + 
            loss_dict = dict()
            loss_dict['total'] = losses
            loss_dict['2d'] = loss_2d
            loss_dict['3d'] = loss_3d
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # scheduler.step() # no scheduler

            # if loss_3d > 0:
            #     optimizer2.zero_grad()
            #     loss_3d.backward()
            #     optimizer2.step()
            #     scheduler2.step()
            # losses = loss_dict['total_loss']

            # for idx_v ,node in enumerate(cfg.DATASET.CAM):
            #     # import pdb; pdb.set_trace()
            #     imgs = images[:,idx_v,...]
            #     val = valids[:,idx_v,...]
            #     lab = labels[:,idx_v,...]
            #     rdep = rdepth[:,idx_v,...]
            #     loss_dict, outputs = model(imgs, val, lab, rdep)
            #     losses = loss_dict['total_loss']

            # optimizer.zero_grad()
            # losses.backward()
            # optimizer.step()
            # scheduler.step()
            
            if cfg.RUN_EFFICIENT: # efficiency
                del matched_pred_single, matched_pred3d, gt_3d, gt_bodys_2d, cam_info

            if engine.local_rank == 0:
                if iteration % 20 == 0 or iteration == max_iter:
                    log_str = 'Iter:%d, LR(1):%.1e, ' % (
                        iteration, optimizer.param_groups[0]["lr"] / num_gpu)
                    for key in loss_dict:
                        tb_writer.add_scalar(
                            key, loss_dict[key].mean(), global_step=iteration)
                        log_str += key + ': %.3f, ' % float(loss_dict[key])

                    time2 = time.time()
                    elapsed_time = time2 - time1
                    time1 = time2
                    required_time = elapsed_time / 20 * (max_iter - iteration)
                    hours = required_time // 3600
                    mins = required_time % 3600 // 60
                    log_str += 'To Finish: %dh%dmin,' % (hours, mins) 

                    logger.info(log_str)

            if iteration % checkpoint_period == 0 or iteration % max_iter == 0:      
                # 给初测试
                # if iteration == checkpoint_period:
                #     mpjpe = generate_3d_point_pairs(model, None, vali_data_loader, cfg, logger, device,
                #             output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result"), total_iter= -1 ) # initial test
                engine.update_iteration(iteration) # update the iteration
                if not engine.distributed or engine.local_rank == 0:
                    engine.save_and_link_checkpoint(cfg.OUTPUT_DIR)
                pose_file = os.path.join('/home/panzhiyu/project/3d_pose/SMAP/model_logs_0821/stage3_root2/validation_result/','stage3_root2_generate_result_test_orig.pkl')
                logger.info(f'Start test for {iteration}')
                mpjpe = easy_mode(pose_file, model ,cfg, logger, device, output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result"))
                model.train() # changed back to model train
                if mpjpe < best_mpjpe:
                    # save the best model    --- dist contains the module
                    engine.save_best_model(os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                    best_mpjpe = mpjpe
                    logger.info(f'The best mpjpe is {best_mpjpe}, saved True')
                else:
                    logger.info(f'The best mpjpe is {best_mpjpe}, saved False')





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

if __name__ == "__main__":
    main()