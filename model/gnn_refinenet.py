import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint
# import sys
# import os.path as osp
# this_dir = osp.dirname(__file__)
# sys.path.insert(0,osp.join(this_dir,'..',''))

from exps.stage3_root2.config import cfg

import numpy as np
import cv2

class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=True, has_relu=True, efficient=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x


class GNN_RefinedNet(nn.Module):  
    def __init__(self, layers_num = 4 ,out_ch = 64, device=torch.device('cpu')):
        super().__init__()
        self.out_ch = out_ch
        self.cross_conv = conv_bn_relu(256, self.out_ch, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=False) # to 64 dim feature
        # guided point thought? 
        self.device = device
        self.CONN = torch.tensor(cfg.DATASET.PAF.VECTOR)
        graph_adjacent = torch.zeros((cfg.DATASET.KEYPOINT.NUM, cfg.DATASET.KEYPOINT.NUM)).to(device)
        graph_adjacent[self.CONN[:,0], self.CONN[:,1]] = 1
        graph_adjacent[self.CONN[:,1], self.CONN[:,0]] = 1
        self.graph_adjacent = graph_adjacent + torch.eye(cfg.DATASET.KEYPOINT.NUM).to(device)
        degree_matrix = torch.sum(self.graph_adjacent, dim=-1)
        degree_matrix_p = torch.pow(degree_matrix, -0.5)
        self.degree_matrix_p = torch.diag(degree_matrix_p)
        self.layers_num = layers_num
        self.gnn_para = nn.ParameterList()
        self.relu = nn.ReLU()
        output_ch = [self.out_ch,64,32,16,3]
        self.laplacian = torch.diag(degree_matrix) - self.graph_adjacent
        self.lg = self.degree_matrix_p @ self.laplacian @ self.degree_matrix_p
        for i in range(self.layers_num):
            w = (torch.rand((cfg.DATASET.KEYPOINT.NUM, output_ch[i], output_ch[i+1]), dtype = torch.float).to(device) - 0.5 ) * 2
            self.gnn_para.append(nn.Parameter(w ,requires_grad=True)) #, mode='fan_in'nn.init.kaiming_normal(w,nonlinearity='relu'), 

    def forward(self, pred_bodys_2d, pre_feature):  # for single person, all the information for single person # TODO: no batch
        kp_nodes_feature = [[] for _ in range(cfg.DATASET.KEYPOINT.NUM)]
        shape_list = np.array(list(cfg.INPUT_SHAPE))/cfg.dataset.STRIDE
        shape_list = shape_list.astype(np.int).tolist()
        shape_list.append(cfg.DATASET.KEYPOINT.NUM)
        background_template = torch.zeros(shape_list)
        
        conv_feature = dict()
        for v, p_feature in pre_feature.items(): # different views
            out_feature = self.cross_conv(p_feature)
            conv_feature[v] = out_feature # sampling the feature and 
            out_feature = out_feature[0,...]
            p_bodys_2d = pred_bodys_2d[v] # 15 x 3
            p_bodys_2d[:,:2] = torch.round(p_bodys_2d[:,:2]/cfg.dataset.STRIDE) 
            temp = p_bodys_2d.view(-1,3)
            temp[temp[:,2]==0,:] = 0 # in_place operation
            _, height, width = out_feature.shape
            new_y = torch.scatter(background_template[:,0,:], dim=0, index=p_bodys_2d[:,1].cpu().unsqueeze(0).to(torch.int64).repeat(height,1), src=torch.ones((height,cfg.DATASET.KEYPOINT.NUM), dtype=torch.float)) # h x k
            new_x = torch.scatter(background_template[0,:,:], dim=0, index=p_bodys_2d[:,0].cpu().unsqueeze(0).to(torch.int64).repeat(width,1), src=torch.ones((width,cfg.DATASET.KEYPOINT.NUM), dtype=torch.float)) # w x k

            new_bg = torch.einsum('ik,jk -> ijk', new_y, new_x)

            # changed back to the np and do the Gaussian process
            numpy_bg = new_bg.numpy()
            process_bg = cv2.GaussianBlur(numpy_bg, (15,15),0)
            process_bg /= np.amax(process_bg)
            process_mask = torch.tensor(process_bg > 0.68) # 1 sigma range in the dim range

            for k in range(cfg.DATASET.KEYPOINT.NUM):
                if p_bodys_2d[k,2] > 0:
                    selected_mask = process_mask[:,:,k:k+1].repeat(1,1,self.out_ch) # w x h x out_ch
                    selected_mask = selected_mask.permute(2,0,1).to(self.device)
                    selected_feature = out_feature * selected_mask # the dim is one # keep the gradient
                    # selected_feature = selected_feature.view(-1,self.out_ch) # keep the orignal memory
                    # do the pooling operation via 
                    pooling_feature = torch.amax(selected_feature, dim=(-1,-2)) # max pooling along the channel dimension
                    kp_nodes_feature[k].append(pooling_feature.unsqueeze(0))

        # average the different views
        
        for k in range(cfg.DATASET.KEYPOINT.NUM):
            if len(kp_nodes_feature[k]) == 0:
                kp_nodes_feature[k] = torch.zeros((1,self.out_ch)).to(device)
                continue
            kp_nodes_feature[k] = torch.cat(kp_nodes_feature[k], dim=0) # different memory field
            kp_nodes_feature[k] = torch.mean(kp_nodes_feature[k], dim=0,keepdim=True)

        kp_nodes_feature = torch.cat(kp_nodes_feature, dim=0) # total k x 64 dims

        input_feature = kp_nodes_feature  # in tensor form
        for i in range(self.layers_num):
            temp_feature = self.lg @ input_feature
            temp_feature = temp_feature[:,None,:]
            fuse_feature = torch.einsum('kij, kjh -> kih', temp_feature, self.gnn_para[i])
            input_feature = self.relu(fuse_feature)
            input_feature = input_feature[:,0,:]

        return kp_nodes_feature, input_feature
                
def main():
    view = [3,6,12,13,23]
    device = torch.device('cuda')
    # create the feature and pose_2d

    feature = dict()
    pred_2d = dict()
    for v in view:
        feature[v] = torch.rand((1,256,512,832)).to(device)
        pred_2d[v] = torch.rand((15,3)).to(device)
        pred_2d[v][:,0] = torch.round(pred_2d[v][:,0] * 831)
        pred_2d[v][:,1] = torch.round(pred_2d[v][:,1] * 511)
        pred_2d[v][:,2] = torch.round(pred_2d[v][:,2])

    GNN_test = GNN_RefinedNet(out_ch = 64, device= device)

    kp_feature = GNN_test(pred_2d, feature)

    return kp_feature           



if __name__ == '__main__':
    view = [3,6,12,13,23]
    device = torch.device('cuda')
    # create the feature and pose_2d

    feature = dict()
    pred_2d = dict()
    for v in view:
        feature[v] = torch.rand((1,256,512,832)).to(device)
        pred_2d[v] = torch.rand((15,3)).to(device)
        pred_2d[v][:,0] = torch.round(pred_2d[v][:,0] * 831)
        pred_2d[v][:,1] = torch.round(pred_2d[v][:,1] * 511)
        pred_2d[v][:,2] = torch.round(pred_2d[v][:,2])
    
    GNN_test = GNN_RefinedNet(out_ch = 64, device= device)
    GNN_test = GNN_test.to(device)

    kp_feature, output_feature = GNN_test(pred_2d, feature)
        
#         # find the sigma
        


        
