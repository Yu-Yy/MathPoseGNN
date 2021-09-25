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
from lib.utils.dataloader import get_train_loader, get_test_loader
from lib.utils.solver import make_lr_scheduler, make_optimizer
from test_ import generate_3d_point_pairs

# for vist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

def main():
    parser = argparse.ArgumentParser()

    with Engine(cfg, custom_parser=parser) as engine:
        logger = engine.setup_log(
            name='train', log_dir=cfg.OUTPUT_DIR, file_name='log.txt')
        args = engine.args
        ensure_dir(cfg.OUTPUT_DIR)

        model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)

        num_gpu = len(engine.devices)
        # import pdb;pdb.set_trace()
        #  default num_gpu: 8, adjust iter settings
        cfg.SOLVER.CHECKPOINT_PERIOD = \
                int(cfg.SOLVER.CHECKPOINT_PERIOD * 8 / num_gpu) # TODO: 8 -> 1 # divide the view num

        # cfg.SOLVER.CHECKPOINT_PERIOD = 10000
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * 8 / num_gpu)

        logger.info(f'The checkpoint period is {cfg.SOLVER.CHECKPOINT_PERIOD }')
        logger.info(f'max_iter period is {cfg.SOLVER.MAX_ITER}')

        optimizer = make_optimizer(cfg, model, num_gpu)
        scheduler = make_lr_scheduler(cfg, optimizer)

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        if engine.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                broadcast_buffers=False)

        # load the pretrained model
        # if cfg.Pretrained:
        #     model_file = '/home/panzhiyu/project/3d_pose/SMAP/SMAP_model.pth'
        #     state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        #     state_dict = state_dict['model']
        #     model.load_state_dict(state_dict)
            
        
        if engine.continue_state_object: 
            engine.restore_checkpoint(is_restore=False)
        else:
            if cfg.MODEL.WEIGHT: # load the pretrained model
                engine.load_checkpoint(cfg.MODEL.WEIGHT, is_restore=False)
                logger.info(f'load the pretrained model from {cfg.MODEL.WEIGHT}')
        # print('load loader')

        train_data_loader = get_train_loader(cfg, num_gpu=num_gpu, is_dist=engine.distributed,
                                       use_augmentation=True, with_mds=cfg.WITH_MDS) # 图片增强， MDS true

        vali_data_loader = get_test_loader(cfg, num_gpu=num_gpu, local_rank=0, stage='test') # TODO : temp comment

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
        epoch = 1
        best_mpjpe = 10000

        for iteration, (images, valids, labels, rdepth, camera_para, joints3d_global, joints3d_local, scale) in enumerate(
                train_data_loader, engine.state.iteration):
            iteration = iteration + 1
            images = images.to(device)
            valids = valids.to(device)
            labels = labels.to(device)
            rdepth = rdepth.to(device)
            for idx_v ,node in enumerate(cfg.DATASET.CAM):
                # import pdb; pdb.set_trace()
                imgs = images[:,idx_v,...]
                val = valids[:,idx_v,...]
                lab = labels[:,idx_v,...]
                rdep = rdepth[:,idx_v,...]
                loss_dict, outputs = model(imgs, val, lab, rdep)
                losses = loss_dict['total_loss']

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                scheduler.step()
            
            if cfg.RUN_EFFICIENT: # efficiency
                del images, valids, labels, losses

            if engine.local_rank == 0:
                if iteration % 20 == 0 or iteration == max_iter:
                    log_str = 'Iter:%d, LR:%.1e, ' % (
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

            if iteration % 5000 == 0:
                # visualization
                folder_name = os.path.join(cfg.OUTPUT_DIR, 'debug_train_pics')
                hm_folder = os.path.join(folder_name,'heatmap')
                paf_folder = os.path.join(folder_name,'paf') # contains the PAF and relative depth map
                joints_folder = os.path.join(folder_name,'joint_3d')
                # uncertainty_folder = os.path.join(folder_name,'uncertainty')
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                if not os.path.exists(hm_folder):
                    os.makedirs(hm_folder)
                if not os.path.exists(paf_folder):
                    os.makedirs(paf_folder)
                if not os.path.exists(joints_folder):
                    os.makedirs(joints_folder)
                
                image = F.interpolate(imgs, scale_factor=0.25)
                vis_pic = tensor2im(image[0,...])
                # get the heatmap, paf, and relative depth map
                heatmap_pred = outputs['heatmap_2d'][-1][-1]
                heatmap_vis_T = heatmap_pred[0]
                heatmap_vis = heatmap_vis_T[:cfg.DATASET.KEYPOINT.NUM,...]
                paf_vis = heatmap_vis_T[cfg.DATASET.KEYPOINT.NUM:,...]
                # rdepth_vis = outputs['det_d'][-1][-1][0]
                # get the ground truth
                labels_vis = lab[0, -1, :, :, :]
                heatmap_gt_vis = labels_vis[:cfg.DATASET.KEYPOINT.NUM,...]
                paf_gt_T = labels_vis[cfg.DATASET.KEYPOINT.NUM:, :, :]
                paf_index = [idx for idx in range(3*cfg.DATASET.PAF.NUM) if idx % 3 != 2]
                paf_gt_vis = paf_gt_T[paf_index, :, :]
                # rdepth_gt_vis = paf_gt_T[2::3, :, :]

                # root depth did not exist the vis map, just show the final joint 
                for hm_idx in range(cfg.DATASET.KEYPOINT.NUM):
                    vis_hm = heatmap_vis[hm_idx,...].clamp(0, 255)\
                                            .byte().detach().cpu().numpy()
                    vis_hm_gt = heatmap_gt_vis[hm_idx,...].clamp(0, 255)\
                                            .byte().detach().cpu().numpy()

                    colored_heatmap_gt = cv2.applyColorMap(vis_hm_gt, cv2.COLORMAP_JET)
                    image_fusing_gt = vis_pic * 0.3 + colored_heatmap_gt * 0.7

                    colored_heatmap = cv2.applyColorMap(vis_hm, cv2.COLORMAP_JET)
                    image_fusing = vis_pic * 0.3 + colored_heatmap * 0.7

                    image_vis = np.concatenate([image_fusing, image_fusing_gt], axis = 1)

                    cv2.imwrite(os.path.join(hm_folder, f'hm_iter_{iteration}_joint_{hm_idx}.jpg'), image_vis)
                for con_idx in range(cfg.DATASET.PAF.NUM):
                    # background = np.zeros((h,w))
                    vis_paf = torch.sum(torch.abs(paf_vis[con_idx*2:con_idx*2+2,...]),dim=0)
                    # .detach().cpu().numpy()
                    vis_bool = vis_paf.clamp(0, 255)\
                                            .byte().detach().cpu().numpy()   #np.sum(np.abs(vis_paf),axis=0) * 2
                    vis_paf_gt = torch.sum(torch.abs(paf_gt_vis[con_idx*2:con_idx*2+2,...]), dim=0) 
                    # .detach().cpu().numpy()
                    vis_bool_gt = vis_paf_gt.clamp(0, 255)\
                                            .byte().detach().cpu().numpy()  #np.sum(np.abs(vis_paf_gt),axis=0) * 2

                    colored_bool_gt = cv2.applyColorMap(vis_bool_gt, cv2.COLORMAP_JET)
                    image_fusing_gt = vis_pic * 0.3 + colored_bool_gt * 0.7

                    colored_bool = cv2.applyColorMap(vis_bool, cv2.COLORMAP_JET)
                    image_fusing = vis_pic * 0.3 + colored_bool * 0.7

                    image_vis = np.concatenate([image_fusing, image_fusing_gt], axis = 1)
                    cv2.imwrite(os.path.join(paf_folder, f'paf_i_{iteration}_joint_{con_idx}.jpg'), image_vis)

            if iteration % max_iter == 0:
                # test on VAl
                mpjpe = generate_3d_point_pairs(model, None, vali_data_loader, cfg, logger, device,
                            output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result"), total_iter=epoch)
                if mpjpe < best_mpjpe:
                    # save the best model    --- dist contains the module
                    engine.save_best_model(os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                    best_mpjpe = mpjpe
                logger.info(f'Finish training  {epoch} process!')
                epoch = epoch + 1
                if iteration >=  4 * max_iter: # 4 个 epoch
                    break

            

        # orig_train
        # for iteration, (images, valids, labels, rdepth) in enumerate(
        #         train_data_loader, engine.state.iteration):
        #     iteration = iteration + 1
        #     images = images.to(device)
        #     valids = valids.to(device)
        #     labels = labels.to(device)
        #     rdepth = rdepth.to(device)
        #     loss_dict, outputs = model(images, valids, labels, rdepth)
        #     # import pdb; pdb.set_trace()

        #     # losses = sum(loss for loss in loss_dict.values())
        #     losses = loss_dict['total_loss']

        #     optimizer.zero_grad()
        #     losses.backward()
        #     optimizer.step()
        #     scheduler.step()

        #     if cfg.RUN_EFFICIENT: # efficiency
        #         del images, valids, labels, losses

        #     if engine.local_rank == 0:
        #         if iteration % 20 == 0 or iteration == max_iter:
        #             log_str = 'Iter:%d, LR:%.1e, ' % (
        #                 iteration, optimizer.param_groups[0]["lr"] / num_gpu)
        #             for key in loss_dict:
        #                 tb_writer.add_scalar(
        #                     key, loss_dict[key].mean(), global_step=iteration)
        #                 log_str += key + ': %.3f, ' % float(loss_dict[key])

        #             time2 = time.time()
        #             elapsed_time = time2 - time1
        #             time1 = time2
        #             required_time = elapsed_time / 20 * (max_iter - iteration)
        #             hours = required_time // 3600
        #             mins = required_time % 3600 // 60
        #             log_str += 'To Finish: %dh%dmin,' % (hours, mins) 

        #             logger.info(log_str)

        #     # do the testing and vis the temporaly result
        #     if iteration % 10000 == 0:
        #         # visualization
        #         folder_name = os.path.join(cfg.OUTPUT_DIR, 'debug_train_pics')
        #         hm_folder = os.path.join(folder_name,'heatmap')
        #         paf_folder = os.path.join(folder_name,'paf') # contains the PAF and relative depth map
        #         joints_folder = os.path.join(folder_name,'joint_3d')
        #         # uncertainty_folder = os.path.join(folder_name,'uncertainty')
        #         if not os.path.exists(folder_name):
        #             os.makedirs(folder_name)
        #         if not os.path.exists(hm_folder):
        #             os.makedirs(hm_folder)
        #         if not os.path.exists(paf_folder):
        #             os.makedirs(paf_folder)
        #         if not os.path.exists(joints_folder):
        #             os.makedirs(joints_folder)
                
        #         image = F.interpolate(images, scale_factor=0.25)
        #         vis_pic = tensor2im(image[0,...])
        #         # get the heatmap, paf, and relative depth map
        #         heatmap_pred = outputs['heatmap_2d'][-1][-1]
        #         heatmap_vis_T = heatmap_pred[0]
        #         heatmap_vis = heatmap_vis_T[:cfg.DATASET.KEYPOINT.NUM,...]
        #         paf_vis = heatmap_vis_T[cfg.DATASET.KEYPOINT.NUM:,...]
        #         # rdepth_vis = outputs['det_d'][-1][-1][0]
        #         # get the ground truth
        #         labels_vis = labels[0, -1, :, :, :]
        #         heatmap_gt_vis = labels_vis[:cfg.DATASET.KEYPOINT.NUM,...]
        #         paf_gt_T = labels_vis[cfg.DATASET.KEYPOINT.NUM:, :, :]
        #         paf_index = [idx for idx in range(3*cfg.DATASET.PAF.NUM) if idx % 3 != 2]
        #         paf_gt_vis = paf_gt_T[paf_index, :, :]
        #         # rdepth_gt_vis = paf_gt_T[2::3, :, :]

        #         # root depth did not exist the vis map, just show the final joint 
        #         for hm_idx in range(cfg.DATASET.KEYPOINT.NUM):
        #             vis_hm = heatmap_vis[hm_idx,...].clamp(0, 255)\
        #                                     .byte().detach().cpu().numpy()
        #             vis_hm_gt = heatmap_gt_vis[hm_idx,...].clamp(0, 255)\
        #                                     .byte().detach().cpu().numpy()

        #             colored_heatmap_gt = cv2.applyColorMap(vis_hm_gt, cv2.COLORMAP_JET)
        #             image_fusing_gt = vis_pic * 0.3 + colored_heatmap_gt * 0.7

        #             colored_heatmap = cv2.applyColorMap(vis_hm, cv2.COLORMAP_JET)
        #             image_fusing = vis_pic * 0.3 + colored_heatmap * 0.7

        #             image_vis = np.concatenate([image_fusing, image_fusing_gt], axis = 1)

        #             cv2.imwrite(os.path.join(hm_folder, f'hm_iter_{iteration}_joint_{hm_idx}.jpg'), image_vis)


        #         for con_idx in range(cfg.DATASET.PAF.NUM):
        #             # background = np.zeros((h,w))
        #             vis_paf = torch.sum(torch.abs(paf_vis[con_idx*2:con_idx*2+2,...]),dim=0)
        #             # .detach().cpu().numpy()
        #             vis_bool = vis_paf.clamp(0, 255)\
        #                                     .byte().detach().cpu().numpy()   #np.sum(np.abs(vis_paf),axis=0) * 2
        #             vis_paf_gt = torch.sum(torch.abs(paf_gt_vis[con_idx*2:con_idx*2+2,...]), dim=0) 
        #             # .detach().cpu().numpy()
        #             vis_bool_gt = vis_paf_gt.clamp(0, 255)\
        #                                     .byte().detach().cpu().numpy()  #np.sum(np.abs(vis_paf_gt),axis=0) * 2

        #             colored_bool_gt = cv2.applyColorMap(vis_bool_gt, cv2.COLORMAP_JET)
        #             image_fusing_gt = vis_pic * 0.3 + colored_bool_gt * 0.7

        #             colored_bool = cv2.applyColorMap(vis_bool, cv2.COLORMAP_JET)
        #             image_fusing = vis_pic * 0.3 + colored_bool * 0.7

        #             image_vis = np.concatenate([image_fusing, image_fusing_gt], axis = 1)
        #             cv2.imwrite(os.path.join(paf_folder, f'paf_i_{iteration}_joint_{con_idx}.jpg'), image_vis)
            
        #     # import pdb; pdb.set_trace()
        #     if iteration % checkpoint_period == 0 or iteration % max_iter == 0:      
        #         # 给初测试
        #         if iteration == checkpoint_period:
        #             mpjpe = generate_3d_point_pairs(model, None, vali_data_loader, cfg, logger, device,
        #                     output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result"), total_iter= -1 ) # initial test
        #         engine.update_iteration(iteration) # update the iteration
        #         if not engine.distributed or engine.local_rank == 0:
        #             engine.save_and_link_checkpoint(cfg.OUTPUT_DIR)

        #     if iteration % max_iter == 0:
        #         # test on VAl
        #         mpjpe = generate_3d_point_pairs(model, None, vali_data_loader, cfg, logger, device,
        #                     output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_result"), total_iter=epoch)
        #         if mpjpe < best_mpjpe:
        #             # save the best model    --- dist contains the module
        #             engine.save_best_model(os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
        #             best_mpjpe = mpjpe
        #         logger.info(f'Finish training  {epoch} process!')
        #         epoch = epoch + 1
        #         if iteration >= 4 * max_iter: # 4 个 epoch
        #             break
            


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
