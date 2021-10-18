import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint
# import sys
# import os.path as osp
# this_dir = osp.dirname(__file__)
# sys.path.insert(0,osp.join(this_dir,'..',''))

from exps.stage3_root2.config import cfg
import torch.nn.functional as F
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


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, device = torch.device('cuda')):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).to(device).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(self.channels,1,1,1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

class GNN_RefinedNet(nn.Module):  # It is a fix trained model
    def __init__(self, layers_num = 4 ,out_ch = 64, device=torch.device('cpu'), max_people = 10):
        super().__init__()
        self.out_ch = out_ch
        self.cross_conv = conv_bn_relu(256, self.out_ch, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=False) # to 64 dim feature
        self.max_people = max_people
        self.gaussion_blur = GaussianBlurConv(max_people * cfg.DATASET.KEYPOINT.NUM).to(device)
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
        w = (torch.rand((3,2), dtype = torch.float).to(device) - 0.5) * 2 
        self.cam_para_encode = nn.Parameter(w, requires_grad=True) # random subspace transformation
        self.layers_num = layers_num
        self.gnn_para = nn.ParameterList()
        self.relu = nn.ReLU()
        output_ch = [self.out_ch * 2,64,32,16,3]
        self.laplacian = torch.diag(degree_matrix) - self.graph_adjacent

        self.lg = self.degree_matrix_p @ self.graph_adjacent @ self.degree_matrix_p #self.laplacian
        for i in range(self.layers_num):
            w = (torch.rand((cfg.DATASET.KEYPOINT.NUM, output_ch[i], output_ch[i+1]), dtype = torch.float).to(device) - 0.5 ) * 2
            self.gnn_para.append(nn.Parameter(w ,requires_grad=True)) #, mode='fan_in'nn.init.kaiming_normal(w,nonlinearity='relu'), 

    def forward(self, pred_bodys_2d, pre_feature, cam_info):  # for single person, all the information for single person # TODO: no batch
        # in batch size (contains the people num)
        # input feature include current feature, current 3D position and projection matrix
        # the input needs to contains the Batch and num_person 
        kp_nodes_feature = [[] for _ in range(cfg.DATASET.KEYPOINT.NUM)] # B X N X DIMS
        shape_list = np.array(list(cfg.INPUT_SHAPE))/cfg.dataset.STRIDE
        shape_list = shape_list.astype(np.int).tolist()
        shape_list.append(cfg.DATASET.KEYPOINT.NUM)
        max_pred = self.max_people
        views = list(pre_feature.keys())
        batch_size = pre_feature[views[0]].shape[0] # TODO: get the batch 
        shape_list.insert(0,max_pred)
        shape_list.insert(0,batch_size)
        background_template = torch.zeros(shape_list)
        conv_feature = dict()

        views_pred_2d = []
        for v, p_feature in pre_feature.items(): # different views  # TODO: add the batch dim 
            cam_p = cam_info[v]
            pro_inv_encode = cam_p['pro_inv'] @ self.cam_para_encode # 4 X 2
            out_feature = self.cross_conv(p_feature)
            conv_feature[v] = out_feature # sampling the feature and # b x c x h x w
            # out_feature = out_feature[0,...] 
            p_bodys_2d = pred_bodys_2d[v] # b x n x 15 x 3
            # find its corresponding 2D pose

            p_bodys_2d[:, :,:,:2] = torch.round(p_bodys_2d[:, :,:,:2]/cfg.dataset.STRIDE) 
            views_pred_2d.append(p_bodys_2d.unsqueeze(0))
            # temp = p_bodys_2d.view(-1,3)
            # # collect the p_bodys_2d
            # temp[temp[:,2]==0,:] = 0 # in_place operation  --> keep all the index available
            _, _, height, width = out_feature.shape
            new_y = torch.scatter(background_template[:,:,:,0,:], dim=2, index=p_bodys_2d[:,:,:,1].cpu().unsqueeze(2).to(torch.int64).repeat(1,1,height,1), src=torch.ones((batch_size, max_pred, height,cfg.DATASET.KEYPOINT.NUM), dtype=torch.float)) # h x k
            new_x = torch.scatter(background_template[:,:,0,:,:], dim=2, index=p_bodys_2d[:,:,:,0].cpu().unsqueeze(2).to(torch.int64).repeat(1,1,width,1), src=torch.ones((batch_size, max_pred ,width,cfg.DATASET.KEYPOINT.NUM), dtype=torch.float)) # w x k

            new_bg = torch.einsum('bnik,bnjk -> bnijk', new_y, new_x)
            new_bg = new_bg.permute(0,1,4,2,3)
            new_bg = new_bg.reshape(batch_size, -1, height, width)
            # new_bg = new_bg.permute(1,2,3,0)
            # # changed back to the np and do the Gaussian process
            # numpy_bg = new_bg.numpy()
            # numpy_bg = numpy_bg.reshape(height, width, -1)
            # it can be get by conv process
            # process_bg = cv2.GaussianBlur(numpy_bg, (5,5),0)
            new_bg = new_bg.to(self.device)
            process_bg = self.gaussion_blur(new_bg)
            process_bg /= torch.amax(process_bg)
            process_mask = (process_bg > 0.1) # 1 sigma range in the dim range 0.68
            process_mask = process_mask.reshape(batch_size, max_pred, cfg.DATASET.KEYPOINT.NUM, height, width)
            process_mask = process_mask.permute(0,1,3,4,2)

            for k in range(cfg.DATASET.KEYPOINT.NUM): #
                # if p_bodys_2d[k,2] > 0:
                selected_mask = process_mask[:,:,:,:,k:k+1].repeat(1,1,1,1,self.out_ch) # w x h x out_ch
                selected_mask = selected_mask.permute(0,1,4,2,3).to(self.device)
                expend_out_feature = out_feature.unsqueeze(1).repeat(1,max_pred,1,1,1)
                selected_feature = expend_out_feature * selected_mask # the dim is one # keep the gradient
                # selected_feature = selected_feature.view(-1,self.out_ch) # keep the orignal memory
                # do the pooling operation via 
                pooling_feature = torch.amax(selected_feature, dim=(-1,-2)) # max pooling along the channel dimension
                # TODO:  process the camera parameters, add one camera para encoder
                pooling_feature = pooling_feature.view(batch_size, max_pred, -1,2)
                cam_encode_feature = torch.einsum('bnij, kj -> bnik', pooling_feature, pro_inv_encode)
                cam_encode_feature = cam_encode_feature.reshape(batch_size, max_pred,-1)
                kp_nodes_feature[k].append(cam_encode_feature.unsqueeze(0))
        views_pred_2d = torch.cat(views_pred_2d, dim=0)
        view_num = views_pred_2d.shape[0]        # average the different views

        for k in range(cfg.DATASET.KEYPOINT.NUM):  # different views fusion by mean
            # if len(kp_nodes_feature[k]) == 0:
            #     kp_nodes_feature[k] = torch.zeros((1,self.out_ch)).to(device)
            #     continue
            valid = views_pred_2d[:,:,:,k,2:3]
            kp_nodes_feature[k] = torch.cat(kp_nodes_feature[k], dim=0) # different memory field
            kp_nodes_feature[k] = kp_nodes_feature[k] * valid # process the invalid view as 0 feature
            valid_views = torch.sum(valid, dim=0)
            # valid_views_t = (valid_views[:, :,0] > 0) # if one kp has no valid points
            valid_views[valid_views == 0] = 0.1 # Inplace operation
            valid_views_inv = 1 / valid_views
            valid_views_count = valid_views_inv.unsqueeze(0).repeat(view_num, 1,1,1)
            kp_nodes_feature[k] = torch.sum(kp_nodes_feature[k] * valid_views_count, dim=0) # It can changed to transformer fusion form
            kp_nodes_feature[k] = kp_nodes_feature[k].unsqueeze(-2)
            # if torch.sum(valid_views_t) == 0:
            #     kp_nodes_feature[k] = torch.zeros((max_pred, 1,self.out_ch)).to(device)
            # else:
            #     kp_nodes_feature[k][valid_views_t,:] = kp_nodes_feature[k][valid_views_t,:] / valid_views[valid_views_t,:] # get the mean
            #     kp_nodes_feature[k][~valid_views_t,:] = torch.zeros((torch.sum(~valid_views_t), self.out_ch)).to(device) # get the zero
            #     kp_nodes_feature[k] = kp_nodes_feature[k].unsqueeze(1)
            # kp_nodes_feature[k] = torch.mean(kp_nodes_feature[k], dim=0,keepdim=True)
        kp_nodes_feature = torch.cat(kp_nodes_feature, dim=-2) # total k x 64 dims
        input_feature = kp_nodes_feature.unsqueeze(-2)  # in tensor form
        for i in range(self.layers_num):
            temp_feature = torch.einsum('ij, bnjok -> bniok', self.lg, input_feature)  #self.lg @ input_feature
            # temp_feature = temp_feature.unsqueeze(-2)
            fuse_feature = torch.einsum('bnkij, kjh -> bnkih', temp_feature, self.gnn_para[i])
            input_feature = self.relu(fuse_feature)
            # input_feature = input_feature[:,0,:]
        output_res = input_feature.squeeze(-2)

        return kp_nodes_feature, output_res
                
def main():
    view = [3,6,12,13,23]
    device = torch.device('cuda')
    # create the feature and pose_2d

    feature = dict()
    pred_2d = dict()
    for v in view:
        feature[v] = torch.rand((1,256,128,208)).to(device)
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
    pred_3d = torch.rand(4,15,4).to(device)
    for v in view:
        feature[v] = torch.rand((1,256,128,208)).to(device)
        pred_2d[v] = torch.rand((4,15,3)).to(device)
        pred_2d[v][:,:,0] = torch.round(pred_2d[v][:,:,0] * 828)
        pred_2d[v][:,:,1] = torch.round(pred_2d[v][:,:,1] * 508)
        pred_2d[v][:,:,2] = torch.round(pred_2d[v][:, :,2])
    
    GNN_test = GNN_RefinedNet(out_ch = 64, device= device)
    GNN_test = GNN_test.to(device)
    kp_feature, output_feature = GNN_test(pred_2d, feature, pred_3d)
        
#         # find the sigma
        


        
