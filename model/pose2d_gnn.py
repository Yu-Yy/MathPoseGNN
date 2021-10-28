import torch
import torch.nn as nn
from exps.stage3_root2.config import cfg
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCNConv
from torch.nn.modules.activation import MultiheadAttention
from torch.nn import Transformer

# batch test
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import time

# class PairData(Data):
#     def __init__(self, edge_index_s=None, x_s=None, edge_index_m=None, x_m=None, edge_weight_s=None, edge_weight_m=None):
#         super().__init__()
#         self.edge_index_s = edge_index_s
#         self.x_s = x_s
#         self.edge_index_m = edge_index_m
#         self.x_m = x_m
#         self.edge_weight_m = edge_weight_m
#         self.edge_weight_s = edge_weight_s
#     def __inc__(self, key, value, *args, **kwargs):
#         if key == 'edge_index_s':
#             return self.x_s.size(0)
#         if key == 'edge_index_m':
#             return self.x_m.size(0)
#         else:
#             return super().__inc__(key, value, *args, **kwargs)
#     def __cat_dim__(self, key, value, *args, **kwargs):
#          if key == 'foo':
#              return None
#          else:
#              return super().__cat_dim__(key, value, *args, **kwargs)

pairs = [[0, 1], [0, 2], [0, 9], [9, 10], [10, 11],
         [0, 3], [3, 4], [4, 5], [2, 12], [12, 13], 
         [13, 14], [2, 6], [6, 7], [7, 8]]
edge_index = torch.tensor(pairs, dtype=torch.long)
edge_index = torch.cat([edge_index, edge_index[:,[1,0]]], dim=0)

# edge_index = torch.tensor([[0, 1, 1, 2],
#                             [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# class GCN(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.conv1 = GCNConv(1, 16)
#         self.conv2 = GCNConv(16, 3)
#         self.bn1 = nn.BatchNorm1d(16)

#     def forward(self, x, edge_index):
#         edge_weight = torch.ones(3,4)
#         x = self.conv1(x, edge_index, edge_weight)
#         x = self.bn1(x.permute(0,2,1))
#         x = x.permute(0,2,1)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)



class Pose_GCN(nn.Module): # This is the regression module
    def __init__(self, inplaneS=7, inplaneM=4, device = torch.device('cuda')):
        # x need to shift and scale (gt is the same) according to the init pose --> to meter metrix representation (cm to m)
        super().__init__()
        self.scalar = 100 # cm to m
        # single view process # init
        self.edge_index = edge_index.t().contiguous()
        self.edge_num = edge_index.shape[0]
        self.device = device
        self.inplaneS = inplaneS
        self.all_layers_num = 3 # TODO: thsi is the hyper parameter
        self.single_view_layers = [inplaneS,32,64,128,256]
        self.single_viewconvlayers = nn.ModuleList()
        self.single_viewbn = nn.ModuleList()
        self.single_viewac = nn.ModuleList()
        for l in range(len(self.single_view_layers) - 1):
            self.single_viewconvlayers.append(GCNConv(self.single_view_layers[l], self.single_view_layers[l+1]))
            self.single_viewbn.append(nn.BatchNorm1d(self.single_view_layers[l+1]))
            self.single_viewac.append(nn.ReLU())
        # multiple view process 
        # self.embed_subspaceQ = nn.Linear(64, 64) # init 3D as q
        # self.embed_subspaceK = nn.Linear(64, 64) # others view as k
        # self.softmax = nn.Softmax(dim=-1) # TODO: adjust 

        # using transformers method to get the fused feature
        self.init3d_layers = [inplaneM, 32, 64, 128, 256] 
        self.init_convlayers = nn.ModuleList()
        self.init_bn = nn.ModuleList()
        self.init_ac = nn.ModuleList()
 
        for l in range(len(self.init3d_layers) - 1):
            self.init_convlayers.append(GCNConv(self.init3d_layers[l],self.init3d_layers[l+1]))
            self.init_bn.append(nn.BatchNorm1d(self.init3d_layers[l+1]))
            self.init_ac.append(nn.ReLU())

        # perform the cross attention along with views and init3d
        self.encode_score = nn.Sequential(nn.Linear(self.single_view_layers[-1]+1,self.single_view_layers[-1]), nn.ReLU())
        self.cross_attention = MultiheadAttention(embed_dim = self.init3d_layers[-1], num_heads = 4, dropout=0.5)
        self.layerNorm_cross = nn.LayerNorm(self.init3d_layers[-1])
        self.self_attention = MultiheadAttention(embed_dim = self.init3d_layers[-1], num_heads = 4, dropout=0.5) # update each view's feature
        self.layerNorm_self = nn.LayerNorm(self.single_view_layers[-1])
        self.single_update = [256, 128, 128, 32, 32]
        self.signle_upconvlayers = nn.ModuleList()
        self.single_upbn = nn.ModuleList()
        self.single_upac = nn.ModuleList()
        for l in range(len(self.single_update) - 1):
            self.signle_upconvlayers.append(GCNConv(self.single_update[l], self.single_update[l+1]))
            self.single_upbn.append(nn.BatchNorm1d(self.single_update[l+1]))
            self.single_upac.append(nn.ReLU())
        self.single_outlayer = nn.Linear(self.single_update[-1], 3)
        
        # After fuse the feature and do the residue add 
        self.final3d_layers = [256, 128, 128, 32, 32]
        self.final_convlayers = nn.ModuleList()
        self.final_bn = nn.ModuleList()
        self.final_ac = nn.ModuleList()
        for l in range(len(self.final3d_layers) - 1):
            self.final_convlayers.append(GCNConv(self.final3d_layers[l], self.final3d_layers[l+1]))
            self.final_bn.append(nn.BatchNorm1d(self.final3d_layers[l+1]))
            self.final_ac.append(nn.ReLU())
        # final output layer
        self.output_layer = nn.Linear(self.final3d_layers[-1], 3)
        
    def forward(self, pose_single, init_3d): #, gt_2d, gt_3d
        # input:
        #     pose_single: dict(), position, direction and paf score
        #     init_3d: tensor, position of the 3d pose, do not update the "0" label joint
        #     gt_3d and gt 2d
        #     cam_info: the projection matrix
        # output:
        #     refined_3d
        #     loss_singleview (with mean) and loss_3d
        batch_size = init_3d.shape[0]
        init_3d_valid = torch.sum(init_3d[:,:,:,3] >0, dim=-1)
        init_3d_pos = init_3d[:,:,:,:3] / self.scalar # unit # B X N X K X 3
        init_3d_pos = init_3d_pos.view(-1, cfg.DATASET.KEYPOINT.NUM, 3)  # batch and people fuse into one new batch
        init_3d_valid = init_3d_valid.view(-1)
        valid_people = (init_3d_valid > 3) #
        # batch_mean = torch.sum(init_3d_pos[valid_people,...] * (1/init_3d_valid[valid_people].unsqueeze(-1).unsqueeze(-1)), dim=-2, keepdim=True) # N, 1, 3
        # set minus the pelvis joints label 
        batch_mean = init_3d_pos[valid_people,2:3,:] # the mid_hip position N,1,3 # 
        init_3d_pos[valid_people,...] = init_3d_pos[valid_people,...] - batch_mean  # N x K x 3
        origin_pos = init_3d_pos.clone()
        new_batch_size = init_3d_pos.shape[0] # BXm
        view_num = len(pose_single) # valid view num
        batch_edge3d_index = torch.cat([edge_index + x*cfg.DATASET.KEYPOINT.NUM for x in range(new_batch_size)], dim=0).to(self.device) 
        batch_edge2d_index = torch.cat([edge_index + x*cfg.DATASET.KEYPOINT.NUM for x in range(new_batch_size * view_num)], dim=0).to(self.device)
        pose_single_view = torch.cat(list(pose_single.values()), dim = 0) # expand the batch size
        # for pose_single_view in pose_single:
        pose_single_view = pose_single_view.reshape(-1,cfg.DATASET.KEYPOINT.NUM, self.inplaneS) # VxBXP x K x 7
        pose_single_view_score = pose_single_view.reshape(view_num, -1, cfg.DATASET.KEYPOINT.NUM, self.inplaneS)
        pose_single_view_score = torch.sum(pose_single_view_score[...,6], dim=-1, keepdim=True)  # V x N x 1 --> V x N x K x 1
        pose_single_view_score = pose_single_view_score[...,None].repeat(1,1,cfg.DATASET.KEYPOINT.NUM,1)
        # process the feature and weight
        single_valid_people = valid_people.repeat(view_num)
        batch_mean_single = batch_mean.repeat(view_num,1,1)
        single_view_feature = pose_single_view.clone() # N x K x 7
        single_view_feature[single_valid_people,:,:3] = (single_view_feature[single_valid_people,:,:3]/ self.scalar) - batch_mean_single
        pose_single_view_bak = single_view_feature.reshape(view_num, -1, cfg.DATASET.KEYPOINT.NUM, self.inplaneS).clone()
        single_view_feature = single_view_feature.reshape(-1,7)
        init_graph_weight = (pose_single_view[...,6] > 0.3)  # just 0 and 1, dropped the zero predict point
        input_graph_weight = init_graph_weight.reshape(-1)
        single_edge_weight = input_graph_weight[batch_edge2d_index[:,0]].float()
        # changed to multiple input
        # single_view_feature = torch.cat([single_view_feature, ], dim=-1)
        # pass the single layer
        for l in range(len(self.single_view_layers)-1):
            pass_single = self.single_viewconvlayers[l](single_view_feature, batch_edge2d_index.t().contiguous())  #, single_edge_weight
            bn_single = pass_single.reshape(-1, cfg.DATASET.KEYPOINT.NUM, self.single_view_layers[l+1]).permute(0,2,1)
            bn_single = self.single_viewbn[l](bn_single)
            ac_single = self.single_viewac[l](bn_single)
            single_view_feature = ac_single.permute(0,2,1).reshape(-1, self.single_view_layers[l+1])
        # single_view_layers is VxBxNxK, 16
        view_feature = single_view_feature.reshape(view_num, -1, cfg.DATASET.KEYPOINT.NUM, self.single_view_layers[-1])
        # add a view score
        view_feature = torch.cat([view_feature, pose_single_view_score], dim=-1) # FC to 128
        encode_feature = self.encode_score(view_feature.reshape(-1, self.single_view_layers[-1]+1))
        # pass the init_layer
        init_3d_pos = init_3d_pos.reshape(-1,3)
        init_3d_weight = (init_3d[:,:,:,3] >0) # BNK
        init_3d_weight = init_3d_weight.reshape(-1)
        init_edge_weight = init_3d_weight[batch_edge3d_index[:,0]].float()
        init_3d_pos = torch.cat([init_3d_pos, init_3d_weight.float().unsqueeze(-1)], dim=-1)
        
        for l in range(len(self.init3d_layers)-1):
            pass_init = self.init_convlayers[l](init_3d_pos, batch_edge3d_index.t().contiguous()) #
            bn_init = pass_init.reshape(-1,cfg.DATASET.KEYPOINT.NUM, self.init3d_layers[l+1]).permute(0,2,1)
            bn_init = self.init_bn[l](bn_init)
            ac_init = self.init_ac[l](bn_init)
            init_3d_pos = ac_init.permute(0,2,1).reshape(-1, self.init3d_layers[l+1])
        init_3d_feature = init_3d_pos.reshape(1, -1, cfg.DATASET.KEYPOINT.NUM, self.init3d_layers[-1])
        # cross attention
        # view_feature is V, N, K ,64; init_3d_feature 1, N, K, 64  -->  1, NK, 64; 
        q_feature = init_3d_feature.reshape(1, -1, self.init3d_layers[-1]) # 1, NK, 64
        t_feature = encode_feature.reshape(view_num, -1, self.single_view_layers[-1]) # V, NK, 64
        # update the t_feature
        v_feature = encode_feature.reshape(view_num, -1, self.single_view_layers[-1])
        new_v_feature = []
        refined_single_pos = []
        for v in range(view_num):
            view_list = list(range(view_num))
            q_single_feature = v_feature[v:v+1,...]
            t_signle_feature = v_feature[view_list.remove(v),...]
            update_feature = self.self_attention(q_single_feature, t_signle_feature, t_signle_feature)[0]
            update_feature = update_feature + q_single_feature
            update_feature = self.layerNorm_self(update_feature)
            new_v_feature.append(update_feature.clone())
            # set the update
            current_pos = pose_single_view_bak[v,:,:,:3]
            update_weight = (pose_single_view_bak[v,:,:,6] > 0.3)
            update_weight = update_weight.reshape(-1)
            update_edge_weight = update_weight[batch_edge3d_index[:,0]].float()
            update_feature = update_feature.reshape(-1, self.single_update[0])
            for l in range(len(self.single_update)-1):
                pass_single_update = self.signle_upconvlayers[l](update_feature, batch_edge3d_index.t().contiguous())
                bn_single_up = pass_single_update.reshape(-1, cfg.DATASET.KEYPOINT.NUM, self.single_update[l+1]).permute(0,2,1)
                bn_single_up = self.single_upbn[l](bn_single_up)
                ac_single_up = self.single_upac[l](bn_single_up)
                update_feature = ac_single_up.permute(0,2,1).reshape(-1, self.single_update[l+1])
            update_feature = update_feature.reshape(-1, cfg.DATASET.KEYPOINT.NUM, self.single_update[-1])
            update_output = self.single_outlayer(update_feature)
            # restrain the residue or not 
            refined_pos_single = current_pos + update_output
            refined_pos_single[valid_people,...] = refined_pos_single[valid_people,...] + batch_mean
            refined_pos_single = refined_pos_single * self.scalar
            refined_pos_single = refined_pos_single.reshape(batch_size, -1, cfg.DATASET.KEYPOINT.NUM, 3)
            refined_single_pos.append(refined_pos_single.unsqueeze(0))
        
        new_v_feature = torch.cat(new_v_feature, dim=0)
        refined_single_pos = torch.cat(refined_single_pos, dim=0)
        new_v_feature = new_v_feature.reshape(view_num, -1, self.single_view_layers[-1])

        fuse_feature = self.cross_attention(q_feature, new_v_feature, new_v_feature)[0] # t_feature
        final_feature = fuse_feature + q_feature
        final_feature = self.layerNorm_cross(final_feature) # 1 NK 64
        final_feature = final_feature.reshape(-1,self.final3d_layers[0])
        # final layer
        for l in range(len(self.final3d_layers)-1):
            pass_final = self.final_convlayers[l](final_feature, batch_edge3d_index.t().contiguous()) #
            bn_final = pass_final.reshape(-1,cfg.DATASET.KEYPOINT.NUM, self.final3d_layers[l+1]).permute(0,2,1)
            bn_final = self.final_bn[l](bn_final)
            ac_final = self.final_ac[l](bn_final)
            final_feature = ac_final.permute(0,2,1).reshape(-1, self.final3d_layers[l+1])
        if torch.any(torch.isnan(final_feature)):
            import pdb;pdb.set_trace()
        final_feature = final_feature.reshape(-1, cfg.DATASET.KEYPOINT.NUM, self.final3d_layers[-1])
        final_output = self.output_layer(final_feature) # N x K x 3
        refined_pose = final_output + origin_pos

        abs_residue = torch.mean(torch.norm(final_output, dim=-1)) # in original metric
        refined_pose[valid_people,...] = refined_pose[valid_people,...] + batch_mean
        refined_pose = refined_pose * self.scalar
        refined_pose = refined_pose.reshape(batch_size, -1, cfg.DATASET.KEYPOINT.NUM, 3)

        return refined_pose, abs_residue, refined_single_pos

if __name__ == '__main__':
    device = torch.device('cuda')
    view_set = [3,6,12,13,23]
    gnn = Pose_GCN(inplaneS=6, inplaneM=3, device=torch.device('cuda'))
    gnn = gnn.to(device)
    pose_single = dict()
    for v in view_set:
        pose_single[v] = torch.rand(4,10,15,7).to(device)
    init_3d = torch.rand(4,10,15,4).to(device)
    start = time.time()
    final_output = gnn(pose_single, init_3d) # pose_single, init_3d, gt_2d, gt_3d
    end = time.time()
    print(f'gnntest time {end - start}')


# class pose_gnn(nn.Module):
#     # use the 2D position and direction
#     def __init__(self, max_people = 10,device=torch.device('cuda')):
#         super().__init__()
#         self.device = device
#         self.max_people = max_people
#         self.CONN = torch.tensor(cfg.DATASET.PAF.VECTOR)
#         graph_adjacent = torch.zeros((cfg.DATASET.KEYPOINT.NUM, cfg.DATASET.KEYPOINT.NUM)).to(device)
#         graph_adjacent[self.CONN[:,0], self.CONN[:,1]] = 1
#         graph_adjacent[self.CONN[:,1], self.CONN[:,0]] = 1
#         self.graph_adjacent = graph_adjacent + torch.eye(cfg.DATASET.KEYPOINT.NUM).to(device)
#         degree_matrix = torch.sum(self.graph_adjacent, dim=-1)
#         degree_matrix_p = torch.pow(degree_matrix, -0.5)
#         self.degree_matrix_p = torch.diag(degree_matrix_p)
#         self.laplacian = torch.diag(degree_matrix) - self.graph_adjacent
        

#     def forward(self, pose_2d, init_3d, cam_info): # with the batch dimension
        
#         print('iii')
        