from lib.utils.post_3d import projectjointsPoints_torch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class optimize_step(nn.Module):
    def __init__(self, reg_p, pred_num, kp_num, device=torch.device('cuda')):
        super().__init__()
        self.reg_p = reg_p
        self.image_size = [1920,1080]
        self.delta_p = nn.Parameter(torch.rand(pred_num,kp_num,3), requires_grad=True)
        # self.delta_p = nn.Parameter(init_para, requires_grad=True) # init para is the init kp
        self.c_pose_c = dict()
        self.project_idx_c = dict()
        self.mask_c = dict()

        # self.optimizer = optim.Adam(self.delta_p, lr = lr)
    
    def forward(self, pred_bodys_3d, pose_2d_collect, cam_info, threshold = 50):
        # match the 2d pose
        # pred_bodys_3d_pos = pred_bodys_3d[:,:,:3]
        # pred_num = pred_bodys_3d_pos.shape[0]
        # kp_num = pred_bodys_3d_pos.shape[1]
        # pred_bodys_3d_vis = (pred_bodys_3d[:,:,3] > 0)

        # delta_pose = self.delta_p[:pred_num, ...]
        # refined_pose_pre = pred_bodys_3d_pos + delta_pose
        # refined_pose = refined_pose_pre.reshape(-1,3)
        # loss = 0

        # for per_2d, cam_p, project_idx in zip(aligned_pose_2d_collect, cam_info, project_idx_collect):
        #     if per_2d is None:
        #         continue
        #     project_refined_2d = projectjointsPoints_torch(refined_pose, cam_p['K'], cam_p['R'], cam_p['t'],cam_p['distCoef'])
        #     project_refined_2d = project_refined_2d.reshape(pred_num,kp_num,2)
        #     c_refined_2d = project_refined_2d[project_idx,...]
        #     vis_pose = (per_2d[...,2] > 0.5)
        #     c_pose_pos = per_2d[:,:,:2]
        #     total_mask = vis_pose * pred_bodys_3d_vis[project_idx,...]
        #     loss = loss + torch.sum((torch.norm(c_refined_2d - c_pose_pos, dim=-1) ** 2) * total_mask)

        # regularization = self.reg_p * (torch.norm(delta_pose) ** 2)
        # loss = loss + regularization

        # refined_pose3d = torch.cat([refined_pose_pre, pred_bodys_3d[:,:,3:4]], dim=-1)
        # return refined_pose3d, loss

        pred_bodys_3d_pos = pred_bodys_3d[:,:,:3]
        pred_num = pred_bodys_3d_pos.shape[0]
        kp_num = pred_bodys_3d_pos.shape[1]
        pred_bodys_3d_vis = (pred_bodys_3d[:,:,3] > 0)

        delta_pose = self.delta_p[:pred_num, ...]
        refined_pose_pre = pred_bodys_3d_pos + delta_pose

        # refined_pose_pre = self.delta_p# for another testing

        refined_pose = refined_pose_pre.reshape(-1,3)
        pred_bodys_3d_r = pred_bodys_3d_pos.reshape(-1,3)
        loss = 0
        if len(self.c_pose_c) == 0:
            # for per_2d, cam_p in zip(pose_2d_collect, cam_info):
            for k, per_2d in pose_2d_collect.items():
                cam_p = cam_info[k]
                if len(per_2d) == 0:
                    self.c_pose_c[k] = None
                    self.project_idx_c[k] = None
                    self.mask_c[k] = None
                    continue
                project_refined_2d = projectjointsPoints_torch(refined_pose, cam_p['K'], cam_p['R'], cam_p['t'],cam_p['distCoef'])
                project_2d = projectjointsPoints_torch(pred_bodys_3d_r, cam_p['K'], cam_p['R'], cam_p['t'],cam_p['distCoef'])
                project_refined_2d = project_refined_2d.reshape(pred_num,kp_num,2)
                project_2d = project_2d.reshape(pred_num,kp_num,2)
                x_check = torch.bitwise_and(project_2d[:, :, 0] >= 0, 
                                            project_2d[:, :, 0] <= self.image_size[0] - 1) #(15,) bool
                y_check = torch.bitwise_and(project_2d[:, :, 1] >= 0,
                                            project_2d[:, :, 1] <= self.image_size[1] - 1) # just fix the coord
                check = torch.bitwise_and(x_check, y_check) # N
                er_match = dict()
                er_id = dict()
                for idx, (p_2d, p_check, vis_3d_p) in enumerate(zip(project_2d, check, pred_bodys_3d_vis)):
                    match_check = p_check * vis_3d_p # valid project2d and valid region
                    temp_er = []
                    if torch.sum(match_check) < 3:  # 
                        continue
                    for off_2d in per_2d:
                        off_2d_pose = off_2d[:,:2]
                        off_2d_vis = (off_2d[:,2] > 0.1) # for the out match 
                        mask = match_check * off_2d_vis
                        if torch.sum(mask) == 0:
                            temp_er.append(1000)
                            continue
                        c_err = torch.mean(torch.norm(off_2d_pose[mask,...] - p_2d[mask,...], dim=-1))
                        temp_er.append(c_err)
                    
                    # judge if it is all err
                    temp_er = torch.tensor(temp_er)
                    if torch.sum(temp_er < threshold) == 0:
                        continue
                    min_gt = torch.argmin(temp_er)
                    min_er = torch.min(temp_er)
                    if min_gt.item() in er_match.keys():
                        if min_er < er_match[min_gt.item()]:
                            er_match[min_gt.item()] = min_er
                            er_id[min_gt.item()] = idx
                    else:
                        er_match[min_gt.item()] = min_er
                        er_id[min_gt.item()] = idx
                
                match_list = list(er_id.keys())
                project_idx = torch.tensor([er_id[v] for v in match_list])  # TODO: confirm the order list(er_id.values())
                match_idx = torch.tensor(match_list)
                
                if len(match_idx) == 0: # no match
                    self.c_pose_c[k] = None
                    self.project_idx_c[k] = None
                    self.mask_c[k] = None
                    continue
                c_pose = per_2d[match_idx,...]
                c_refined_2d = project_refined_2d[project_idx,...] # 借用匹配结果
                self.project_idx_c[k] = project_idx
                check = check[project_idx,...]
                vis_pose = (c_pose[...,2] > 0.3) # do not consider the low and false views
                c_pose_pos = c_pose[:,:,:2]
                self.c_pose_c[k] = c_pose_pos
                total_mask = check * vis_pose * pred_bodys_3d_vis[project_idx,...]
                self.mask_c[k] = total_mask
                loss = loss + torch.sum((torch.norm(c_refined_2d - c_pose_pos, dim=-1) ** 2) * total_mask)
        else:
            # for per_2d, cam_p, project_idx, total_mask in zip(self.c_pose_c, cam_info, self.project_idx_c, self.mask_c):
            for k, per_2d in self.c_pose_c.items():
                cam_p = cam_info[k]; project_idx = self.project_idx_c[k]; total_mask = self.mask_c[k]         
                if per_2d is None:
                    continue
                project_refined_2d = projectjointsPoints_torch(refined_pose, cam_p['K'], cam_p['R'], cam_p['t'],cam_p['distCoef'])
                project_refined_2d = project_refined_2d.reshape(pred_num,kp_num,2)
                c_refined_2d = project_refined_2d[project_idx,...]
                loss = loss + torch.sum((torch.norm(c_refined_2d - per_2d, dim=-1) ** 2) * total_mask)

        regularization = self.reg_p * (torch.norm(delta_pose) ** 2)
        # regularization = self.reg_p * (torch.norm(pred_bodys_3d_pos - refined_pose_pre) ** 2)
        
        loss = loss + regularization
        
        refined_pose3d = torch.cat([refined_pose_pre, pred_bodys_3d[:,:,3:4]], dim=-1)
        return refined_pose3d, loss


# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
