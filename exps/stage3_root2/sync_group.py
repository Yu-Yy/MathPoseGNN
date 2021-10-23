from cv2 import norm, threshold
from numpy.core.fromnumeric import repeat
from numpy.lib.function_base import extract
import torch
import numpy as np
import pickle
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from torch.distributed.distributed_c10d import group, new_group
# from pointnet2_pro.pointnet2_modules import PointnetSAModuleDebug, PointnetFPModule 
from exps.stage3_root2.pointnet2_pro.pointnet2_modules import PointnetSAModuleDebug, PointnetFPModule 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import scipy.io
import time
import itertools
from lib.utils.post_3d import projectjointsPoints_torch



device = torch.device('cuda')
# from config import cfg
from exps.stage3_root2.config import cfg
joint_to_limb_heatmap_relationship = cfg.DATASET.PAF.VECTOR
paf_z_coords_per_limb = list(range(cfg.DATASET.KEYPOINT.NUM))
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)


class PR_FPS:
    def __init__(self, points):
        self.points = np.unique(points, axis=0)
        self.probalistic(self.points)

    def probalistic(self,points):
        # di_pointcloud with N x 6
        mu = points[:,:3]
        va_direction = points[:,3:]
        orth1 = np.ones(va_direction.shape)
        orth1[:,2] = -(va_direction[:,0] + va_direction[:,1]) /  va_direction[:,2]
        orth1 = orth1 / np.linalg.norm(orth1,axis=-1,keepdims=True)
        orth2 = np.cross(va_direction, orth1)
        # get the covariance matrix
        p = 10 * np.einsum('ij,ik-> ijk', va_direction, va_direction) + np.einsum('ij,ik-> ijk', orth1, orth1) + np.einsum('ij,ik-> ijk', orth2, orth2)
        self.point_mean = mu
        self.points_var = p
        # return mu, p

    def KL_Div(self, gaussian1, gaussian2):
        C1, C2 = gaussian1.C, gaussian2.C
        mu1, mu2 = gaussian1.mu, gaussian2.mu
        #print(torch.inverse(C2))
        KL_1to2 = 0.5 * ( torch.log(torch.det(C2) / torch.det(C1)) - 3 + \
            torch.trace(torch.mm(torch.inverse(C2),C1)) + torch.mm(torch.mm((mu2 - mu1).t(), torch.inverse(C2)), mu2 - mu1) )
        KL_2to1 = 0.5 * ( torch.log(torch.det(C1) / torch.det(C2)) - 3 + \
            torch.trace(torch.mm(torch.inverse(C1),C2)) + torch.mm(torch.mm((mu1 - mu2).t(), torch.inverse(C1)), mu1 - mu2) )
        return (KL_1to2 + KL_2to1) / 2

        

    def get_min_distance(self, a, b):
        distance = []
        for i in range(a.shape[0]):
            dis = np.sum(np.square(a[i] - b), axis=-1)
            distance.append(dis)
        
        distance = np.stack(distance, axis=-1)
        distance = np.min(distance, axis=-1)
        return np.argmax(distance)
 
    def get_min_distance_prob(self, a, b):
        distance = []
        for i in range(a.shape[0]):
            # get the distance from a[i] with b
            distance = self.KL_Div(a[i], b)
        pass

    @staticmethod
    def get_model_corners(model):
        min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
        min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
        min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        return corners_3d
 
    def compute_fps(self, K):
        # 计算中心点位
        corner_3d = self.get_model_corners(self.points)
        center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
        A = np.array([center_3d]) 
        B = np.array(self.points)
        t = []
        # 寻找Ｋ个节点
        for i in range(K):
            max_id = self.get_min_distance(A, B)
            A = np.append(A, np.array([B[max_id]]), 0)
            B = np.delete(B, max_id, 0)
            t.append(max_id)
        # get the corresponding clusterred index  
        print(A)
        ax = plt.axes(projection='3d')
        ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], cmap='Greens')
        plt.show()

def get_model_corners(batch_points,device=torch.device('cuda')):
    # batch_points in tensor
    batch_num = batch_points.shape[0]
    min_x, max_x = torch.min(batch_points[:,:,0],dim=1,keepdim=True)[0], torch.max(batch_points[:,:,0],dim=1,keepdim=True)[0]
    min_y, max_y = torch.min(batch_points[:,:,1],dim=1,keepdim=True)[0], torch.max(batch_points[:,:,1],dim=1,keepdim=True)[0]
    min_z, max_z = torch.min(batch_points[:,:,2],dim=1,keepdim=True)[0], torch.max(batch_points[:,:,2],dim=1,keepdim=True)[0]
    corners_3d = torch.cat([torch.cat([min_x, min_y, min_z],dim=1).unsqueeze(1),
                            torch.cat([min_x, min_y, max_z],dim=1).unsqueeze(1),
                            torch.cat([min_x, max_y, min_z],dim=1).unsqueeze(1), 
                            torch.cat([min_x, max_y, max_z],dim=1).unsqueeze(1),
                            torch.cat([max_x, min_y, min_z],dim=1).unsqueeze(1),
                            torch.cat([max_x, min_y, max_z],dim=1).unsqueeze(1),
                            torch.cat([max_x, max_y, min_z],dim=1).unsqueeze(1),
                            torch.cat([max_x, max_y, max_z],dim=1).unsqueeze(1)],
                            dim=1)
    center_point = torch.mean(corners_3d,dim=1,keepdim=True)
    sigma_unit = torch.eye(3)[None,None,...].repeat(batch_num,1,1,1).to(device).reshape(batch_num,1,9)
    center_data = torch.cat([center_point,sigma_unit,sigma_unit],dim=-1)
    cat_xyz = torch.cat([center_data, batch_points],dim=1)
    return cat_xyz


def BKL_Div(mu1,mu2,C1,C2):#gaussian1, gaussian2
    #mu1:[B,N,3]
    #mu2:[B,N,3]
    #C1:[B,N,3,3]
    #C2:[B,N,3,3]
    # C1 = gaussian1[:,:,3:12].reshape(1,-1,3,3)
    # C2 = gaussian2[:,:,3:12].reshape(1,-1,3,3)
    # mu1 = gaussian1[:,:,:3].reshape(1,-1,3) 
    # mu2 = gaussian2[:,:,:3].reshape(1,-1,3)

    # import pdb; pdb.set_trace()
    N, _ = mu1.shape
    #assert  
    C1 = C1.view(-1,3,3)
    C2 = C2.view(-1,3,3)
    mu1 = mu1.view(-1,3).unsqueeze(1) #[B*N,1,3]
    mu2 = mu2.view(-1,3).unsqueeze(1) #[B*N,1,3]
    trace_mat = torch.eye(3).unsqueeze(0).repeat([N,1,1]).to(mu1.device) #B*
    #print(trace_mat.shape)
    #print(torch.matmul(C2.inverse(), C1).shape)
    KL_1to2 = torch.log(C2.det() / C1.det()) - 3 + torch.sum(torch.matmul(C2.inverse(), C1) * trace_mat, dim = (1,2)) + \
     torch.matmul(torch.matmul(mu2 - mu1, C2.inverse()), (mu2 - mu1).permute(0,2,1) ).squeeze()
    KL_2to1 = torch.log(C1.det() / C2.det()) - 3 + torch.sum(torch.matmul(C1.inverse(), C2) * trace_mat, dim = (1,2)) + \
     torch.matmul(torch.matmul(mu1 - mu2, C1.inverse()), (mu1 - mu2).permute(0,2,1) ).squeeze()
    Div = (KL_1to2 + KL_2to1) / 4
    return Div #.view(B,N)


def KL_Div(gaussian1, gaussian2):
    C1 = gaussian1[3:12].reshape(3,3)
    C2 = gaussian2[3:12].reshape(3,3)
    mu1 = gaussian1[:3].reshape(3,1) 
    mu2 = gaussian2[:3].reshape(3,1)
    KL_1to2 = 0.5 * ( torch.log(torch.det(C2) / torch.det(C1)) - 3 + \
        torch.trace(torch.mm(torch.inverse(C2),C1)) + torch.mm(torch.mm((mu2 - mu1).t(), torch.inverse(C2)), mu2 - mu1) )
    KL_2to1 = 0.5 * ( torch.log(torch.det(C1) / torch.det(C2)) - 3 + \
        torch.trace(torch.mm(torch.inverse(C1),C2)) + torch.mm(torch.mm((mu1 - mu2).t(), torch.inverse(C1)), mu1 - mu2) )
    return (KL_1to2 + KL_2to1) / 2


def probalistic(di_pointcloud):
    # di_pointcloud with B x N x 6
    mu = di_pointcloud[:,:,:3]
    va_direction = di_pointcloud[:,:,3:]
    orth1 = np.ones(va_direction.shape)
    orth1[:,:,2] = -(va_direction[:,:,0] + va_direction[:,:,1]) /  va_direction[:,:,2]
    orth1 = orth1 / np.linalg.norm(orth1,axis=-1,keepdims=True)
    orth2 = np.cross(va_direction, orth1)
    # get the covariance matrix
    p = 9 * np.einsum('bij,bik-> bijk', va_direction, va_direction) + 1 * np.einsum('bij,bik-> bijk', orth1, orth1) + 1 * np.einsum('bij,bik-> bijk', orth2, orth2)
    return mu, p

def probalistic_torch(di_pointcloud):
    # di_pointcloud with B x N x 6
    mu = di_pointcloud[:,:,:3]
    va_direction = di_pointcloud[:,:,3:]
    orth1 = torch.ones(va_direction.shape, dtype= torch.float).to(device)
    orth1[:,:,2] = -(va_direction[:,:,0] + va_direction[:,:,1]) /  va_direction[:,:,2]
    # orth1 = orth1 / np.linalg.norm(orth1,axis=-1,keepdims=True)
    orth1 = orth1 / torch.norm(orth1,dim=-1,keepdim=True)
    orth2 = torch.cross(va_direction, orth1)
    # get the covariance matrix
    p = 9 * torch.einsum('bij,bik-> bijk', va_direction, va_direction) + 1 * torch.einsum('bij,bik-> bijk', orth1, orth1) + 1 * torch.einsum('bij,bik-> bijk', orth2, orth2) #(9 1 1)
    return mu, p

def group_probalistic(orig_xyz, group_inds, group_mask):
    # orig: B, N, 21
    # group: B,Npoint,M
    
    # using unique acoording to different 
    B, npoints, nsamples = group_inds.shape
    dim_n = orig_xyz.shape[-1]
    orig_xyz_cp = orig_xyz[:,None,:,:].repeat(1,npoints,1,1)
    group_inds_tools = group_inds[:,:,:,None].repeat(1,1,1,dim_n)
    extracted = torch.gather(orig_xyz_cp,dim=2,index=group_inds_tools.type(torch.int64))
    extracted_sigmai = extracted[:,:,:,12:].reshape(B,npoints,nsamples,3,3)
    extracted_mu = extracted[:,:,:,:3].reshape(B,npoints,nsamples,3,1)
    group_sigmai = torch.sum(extracted[:,:,:,12:] * (group_mask[...,None].repeat(1,1,1,9)), dim=2).reshape(B,npoints,3,3)
    # get the det value
    # compare_det = torch.linalg.det(group_sigmai) # time costing
    # singular check
    # record the position of 0 point
    add_count = torch.sum(group_mask, dim=-1,keepdim=True)
    add_count = add_count[...,None].repeat(1,1,3,3)
    detect_det = torch.det(group_sigmai)
    vali_mask = (detect_det != 0)
    # import pdb; pdb.set_trace()
    index_mask = vali_mask[...,None,None].repeat(1,1,3,3)
    vali_mask = vali_mask[...,None]
    out_groupi = group_sigmai.clone()
    
    if torch.sum(~vali_mask) > 0: # 
        group_sigmai[~index_mask] = torch.rand(group_sigmai[~index_mask].shape).to(device)
    try:
        group_sigma = torch.inverse(group_sigmai/add_count)/add_count  # (avoid singular) # inverse is time costing
    except:
        import pdb; pdb.set_trace()
        # group_sigmai = group_sigmai +  0.00001 * torch.eye(3).to(device)
        # group_sigma = torch.inverse(group_sigmai)
    # group_sigma = torch.linalg.inv(group_sigmai)
    group_mu = torch.einsum('bnji,bnik->bnjk',group_sigma, torch.sum(torch.einsum('bnmji,bnmik->bnmjk', extracted_sigmai, extracted_mu) * (group_mask[...,None,None].repeat(1,1,1,3,1)),dim=2)) # error
    group_mu = group_mu.squeeze(-1)
    group_data = torch.cat([group_mu[:,1:,:],group_sigma[:,1:,...].flatten(-2),group_sigmai[:,1:,...].flatten(-2)], dim=-1)
    return group_mu, group_sigma, group_data, vali_mask

def p_nms(candis, thresh=torch.tensor(15)): # unit is cm
    '''
    the candis B x N x 3
    '''
    # resort the candis via det
    B,N,_ = candis.shape
    candis_cn = candis[:,:,None,...].repeat(1,1,N,1)
    candis_cn2 = candis[:,None,:,...].repeat(1,N,1,1)
    dist = torch.norm((candis_cn - candis_cn2),dim=-1) # 
    triu_mask =  torch.triu(torch.ones(B,N,N,dtype=torch.bool),diagonal=0).to(dist.device)
    thresh = thresh.to(dist.dtype).to(dist.device)
    dist = torch.where(triu_mask, thresh+1, dist)
    nms_mask = ~torch.sum(dist < thresh , dim=-1, dtype = torch.bool)
    return nms_mask

def unique(candis, thresh=torch.tensor(0.5)): # unit is cm
    '''
    the candis B x N x 3
    '''
    # resort the candis via det
    B,N,_ = candis.shape
    candis_cn = candis[:,:,None,...].repeat(1,1,N,1)
    candis_cn2 = candis[:,None,:,...].repeat(1,N,1,1)
    dist = torch.norm((candis_cn - candis_cn2),dim=-1) # 
    triu_mask =  torch.triu(torch.ones(B,N,N,dtype=torch.bool),diagonal=0).to(dist.device)
    thresh = thresh.to(dist.dtype).to(dist.device)
    dist = torch.where(triu_mask, thresh+1, dist)
    nms_mask = ~torch.sum(dist < thresh , dim=-1, dtype = torch.bool)
    return nms_mask

def main():
    with open(f'pointcloud_kpdebug.pkl','rb') as dfile:
        pc_kp = pickle.load(dfile)
    with open(f'jointgt_debug.pkl','rb') as dfile:
        gt_kp = pickle.load(dfile)
    device = torch.device('cuda')
    # gap 
    gap_tensor = torch.rand(1,11,3,3).to(device)
    gap_inverse = torch.inverse(gap_tensor)
    gap_det = torch.det(gap_tensor)
    # for ep in range(2):
    start = time.time()
    kp_extracted = gt_kp[1,:4,13]
    demo_extracted = pc_kp[13][1]
    mu, sigma = probalistic(demo_extracted[None,...])
    sigma_v = np.linalg.inv(sigma)
    # generate the 21 dim vector
    sigma_f = sigma.reshape(1,-1,9)
    sigma_vf = sigma_v.reshape(1,-1,9)
    
    xyz = np.concatenate([mu, sigma_f, sigma_vf], axis = -1)
    # in tensor
    xyz_tensor = torch.tensor(xyz, dtype=torch.float).to(device) # slowly initially
    # xyz_tensor = xyz_tensor[None,...]
    

    nsample_1 = 512
    PA_FPSTEST = PointnetSAModuleDebug(npoint=10+1,  # mainly for downsampling
                radius=13,
                nsample=nsample_1,
                mlp=[21, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True)


        # nsample_2 = 64
        # PA_FPSTEST2 = PointnetSAModuleDebug(npoint=16+1 ,
        #             radius=80,
        #             nsample=nsample_2,
        #             mlp=[21, 64, 64, 128],
        #             use_xyz=True,
        #             normalize_xyz=True)

        # nsample_3 = 16
        # PA_FPSTEST3 = PointnetSAModuleDebug(npoint=32+1 ,
        #             radius=3,
        #             nsample=nsample_3,
        #             mlp=[21, 64, 64, 128],
        #             use_xyz=True,
        #             normalize_xyz=True)

        # nsample_4 = 8
        # PA_FPSTEST4 = PointnetSAModuleDebug(npoint=16+1 ,
        #             radius=3.5,
        #             nsample=nsample_4,
        #             mlp=[21, 64, 64, 128],
        #             use_xyz=True,
        #             normalize_xyz=True)

    # group 1
    cat_xyz = get_model_corners(xyz_tensor)
    inds, group_inds, debug_xyz1 = PA_FPSTEST(cat_xyz)
    mask = ((group_inds[:,:,0:1].repeat(1,1,nsample_1) - group_inds) !=0)
    mask[:,:,0] = True
    group_mu, group_sigma, xyz2  = group_probalistic(cat_xyz, group_inds, mask) # First point is omitted
    # end2 = time.time()
    # print(f'{end2 - end1}')
    flag = torch.sum(mask,dim=-1)

    # NMS 
    # threshold = 4 # 4cm 
    # pred_bak = group_mu[:,1:,...] # TODO: do the nms in distance

    # group 2
    # xyz2 = next_xyz1[:,1:,:].clone() # 
    # cat_xyz2 = get_model_corners(xyz2)
    # inds2, group_inds2, debug_xyz2 = PA_FPSTEST2(cat_xyz2)
    # mask2 = ((group_inds2[:,:,0:1].repeat(1,1,nsample_2) - group_inds2) !=0)
    # mask2[:,:,0] = True
    # group_mu2, group_sigma2, xyz3  = group_probalistic(cat_xyz2, group_inds2, mask2) # First point is omitted
    
    # print(flag)

    # print(kp_extracted)

    # cat_xyz3 = get_model_corners(xyz3)
    # inds3, group_inds3, _ = PA_FPSTEST3(cat_xyz3)
    # mask3 = ((group_inds3[:,:,0:1].repeat(1,1,nsample_3) - group_inds3) !=0)
    # mask3[:,:,0] = True
    # group_mu3, group_sigma3, xyz4  = group_probalistic(cat_xyz3, group_inds3, mask3) # First point is omitted

    # cat_xyz4 = get_model_corners(xyz4)
    # inds4, group_inds4, _ = PA_FPSTEST4(cat_xyz4)
    # mask4 = ((group_inds4[:,:,0:1].repeat(1,1,nsample_4) - group_inds4) !=0)
    # mask4[:,:,0] = True
    # group_mu4, group_sigma4, xyz5  = group_probalistic(cat_xyz4, group_inds4, mask4) # First point is omitted


    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(121,projection='3d') #
    ax.scatter(xyz[0,:,0], xyz[0,:,1], xyz[0,:,2], marker='o', s = 4)
    ax.scatter(group_mu[0,:,0].cpu().numpy(), group_mu[0,:,1].cpu().numpy(), group_mu[0,:,2].cpu().numpy(), marker='^',s = 320,c='k')
    ax.scatter(kp_extracted[:,0].cpu().numpy(), kp_extracted[:,1].cpu().numpy(), kp_extracted[:,2].cpu().numpy(),marker='^',s=640)

    ax = fig.add_subplot(122,projection='3d')
    # ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], marker='o', s = 4)
    # ax.scatter(group_mu2[0,:,0].cpu().numpy(), group_mu2[0,:,1].cpu().numpy(), group_mu2[0,:,2].cpu().numpy(), marker='^',s = 16,c='k')
    # ax.scatter(kp_extracted[:,0].cpu().numpy(), kp_extracted[:,1].cpu().numpy(), kp_extracted[:,2].cpu().numpy(),marker='^',s=640)


    extracted_xyz = cat_xyz[0,inds[0,5].long(),:]
    debug_extracted_xyz = cat_xyz[0,group_inds[0,5].long(),:]
    ax.scatter(xyz[0,:,0], xyz[0,:,1], xyz[0,:,2], marker='o', s = 1, c='g')
    ax.scatter(debug_extracted_xyz[:,0].cpu().numpy(),debug_extracted_xyz[:,1].cpu().numpy(),debug_extracted_xyz[:,2].cpu().numpy(),marker='o', s = 64, c='k')
    ax.scatter(extracted_xyz[0].cpu().numpy(), extracted_xyz[1].cpu().numpy(), extracted_xyz[2].cpu().numpy(), marker='^',s = 640,c='b')


    # debug_extracted_xyz = cat_xyz[0,group_inds[0,4,:300].long(),:].cpu().numpy().tolist()
    # scipy.io.savemat('debug_sample.mat',mdict={'sampling':debug_extracted_xyz})
    
    # with open('debug_sample.pkl','wb') as f:
    #     pickle.dump(debug_extracted_xyz,f)

    plt.savefig('debugfps.png')
    import pdb; pdb.set_trace()



def main2():
    with open(f'pointcloud_kpdebug.pkl','rb') as dfile:
        pc_kp = pickle.load(dfile)
    with open(f'pointcloud_kpdebug_tc.pkl','rb') as dfile:
        pc_kp_tc = pickle.load(dfile)
    with open(f'jointgt_debug.pkl','rb') as dfile:
        gt_kp = pickle.load(dfile)
    kp_extracted = gt_kp[0,:4,8]
    print(kp_extracted)
    demo_extracted = pc_kp_tc[8][0].float()
    
    mu, sigma = probalistic_torch(demo_extracted[None,...])
    # import pdb; pdb.set_trace()
    # sigma_v = np.linalg.inv(sigma)
    sigma_v = torch.inverse(sigma)
    # generate the 21 dim vector
    sigma_f = sigma.reshape(1,-1,9)
    sigma_vf = sigma_v.reshape(1,-1,9)
    device = torch.device('cuda')

    # xyz = np.concatenate([mu, sigma_f, sigma_vf], axis = -1)
    xyz_tensor = torch.cat([mu, sigma_f, sigma_vf], dim = -1) #.to(device)
    xyz = xyz_tensor.cpu().numpy()
    # in tensor
    # xyz_tensor = torch.tensor(xyz, dtype=torch.float).to(device)
    # xyz_tensor = xyz_tensor[None,...]
    
    nsample_1 = 128
    PA_FPSTEST = PointnetSAModuleDebug(npoint=128+1,  # mainly for downsampling
                radius=2,
                nsample=nsample_1,
                mlp=[21, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True)
                
    nsample_2 = 64
    PA_FPSTEST2 = PointnetSAModuleDebug(npoint=64+1 ,
                radius=10,
                nsample=nsample_2,
                mlp=[21, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True)

    nsample_3 = 32
    PA_FPSTEST3 = PointnetSAModuleDebug(npoint=32+1 ,
                radius=100,
                nsample=nsample_3,
                mlp=[21, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True)

    nsample_4 = 16
    PA_FPSTEST4 = PointnetSAModuleDebug(npoint=8+1 ,
                radius=600,
                nsample=nsample_4,
                mlp=[21, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True)


    # group 1 
    cat_xyz = get_model_corners(xyz_tensor)
    inds, group_inds, debug_xyz1 = PA_FPSTEST(cat_xyz)
    mask = ((group_inds[:,:,0:1].repeat(1,1,nsample_1) - group_inds) !=0)
    mask[:,:,0] = True
    group_mu, group_sigma, xyz2  = group_probalistic(cat_xyz, group_inds, mask) # First point is omitted
    flag = torch.sum(mask,dim=-1)
    print(flag)
    # NMS 
    # threshold = 4 # 4cm 
    # pred_bak = group_mu[:,1:,...] # TODO: do the nms in distance




    # group 2 
    cat_xyz2 = get_model_corners(xyz2)
    inds2, group_inds2, debug_xyz2 = PA_FPSTEST2(cat_xyz2)
    mask2 = ((group_inds2[:,:,0:1].repeat(1,1,nsample_2) - group_inds2) !=0)
    mask2[:,:,0] = True
    group_mu2, group_sigma2, xyz3  = group_probalistic(cat_xyz2, group_inds2, mask2) # First point is omitted
    flag2 = torch.sum(mask2,dim=-1)
    print(flag2)
    

    cat_xyz3 = get_model_corners(xyz3)
    inds3, group_inds3, _ = PA_FPSTEST3(cat_xyz3)
    mask3 = ((group_inds3[:,:,0:1].repeat(1,1,nsample_3) - group_inds3) !=0)
    mask3[:,:,0] = True
    group_mu3, group_sigma3, xyz4  = group_probalistic(cat_xyz3, group_inds3, mask3) # First point is omitted
    flag3 = torch.sum(mask3,dim=-1)
    print(flag3)

    cat_xyz4 = get_model_corners(xyz4)
    inds4, group_inds4, _ = PA_FPSTEST4(cat_xyz4)
    mask4 = ((group_inds4[:,:,0:1].repeat(1,1,nsample_4) - group_inds4) !=0)
    mask4[:,:,0] = True
    group_mu4, group_sigma4, xyz5  = group_probalistic(cat_xyz4, group_inds4, mask4) # First point is omitted
    flag4 = torch.sum(mask4,dim=-1)
    # remove the false positive
    print(flag4)
    # NMS for points
    threshold = 20
    final_det = torch.log(1 / torch.det(group_sigma4))
    det_mask = (final_det[:,1:] > threshold)
    nms_mask = p_nms(group_mu4[:,1:,:]) # delete the init one
    select_mask = det_mask * nms_mask
    group_mu4 = group_mu4[:,1:,:]


    final_pred = group_mu4[0,select_mask[0,:],:]
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(121,projection='3d') #
    ax.scatter(xyz[0,:,0], xyz[0,:,1], xyz[0,:,2], marker='o', s = 4)
    # ax.scatter(group_mu4[0,1:,0].cpu().numpy(), group_mu4[0,1:,1].cpu().numpy(), group_mu4[0,1:,2].cpu().numpy(), marker='^',s = 320,c='k')
    ax.scatter(final_pred[:,0].cpu().numpy(), final_pred[:,1].cpu().numpy(), final_pred[:,2].cpu().numpy(), marker='^',s = 320,c='k')
    ax.scatter(kp_extracted[:,0].cpu().numpy(), kp_extracted[:,1].cpu().numpy(), kp_extracted[:,2].cpu().numpy(),marker='^',s=640)

    ax = fig.add_subplot(122,projection='3d')
    # ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], marker='o', s = 4)
    # ax.scatter(group_mu2[0,:,0].cpu().numpy(), group_mu2[0,:,1].cpu().numpy(), group_mu2[0,:,2].cpu().numpy(), marker='^',s = 16,c='k')
    # ax.scatter(kp_extracted[:,0].cpu().numpy(), kp_extracted[:,1].cpu().numpy(), kp_extracted[:,2].cpu().numpy(),marker='^',s=640)


    extracted_xyz = cat_xyz4[0,inds4[0,5].long(),:]
    debug_extracted_xyz = cat_xyz4[0,group_inds4[0,5].long(),:]
    ax.scatter(xyz[0,:,0], xyz[0,:,1], xyz[0,:,2], marker='o', s = 1, c='g')
    ax.scatter(debug_extracted_xyz[:,0].cpu().numpy(),debug_extracted_xyz[:,1].cpu().numpy(),debug_extracted_xyz[:,2].cpu().numpy(),marker='o', s = 64, c='k')
    ax.scatter(extracted_xyz[0].cpu().numpy(), extracted_xyz[1].cpu().numpy(), extracted_xyz[2].cpu().numpy(), marker='^',s = 640,c='b')


    # debug_extracted_xyz = cat_xyz[0,group_inds[0,4,:300].long(),:].cpu().numpy().tolist()
    # scipy.io.savemat('debug_sample.mat',mdict={'sampling':debug_extracted_xyz})
    # with open('debug_sample.pkl','wb') as f:
    #     pickle.dump(debug_extracted_xyz,f)

    plt.savefig('debugfps.png')
    import pdb; pdb.set_trace()
    
    
def pc_fused(batch_pc):
    # with open(f'pointcloud_kpdebug.pkl','rb') as dfile:
    #     pc_kp = pickle.load(dfile)
    # with open(f'pointcloud_kpdebug_tc.pkl','rb') as dfile:
    #     pc_kp_tc = pickle.load(dfile)
    # with open(f'jointgt_debug.pkl','rb') as dfile:
    #     gt_kp = pickle.load(dfile)
    # kp_extracted = gt_kp[0,:4,8]
    # print(kp_extracted)
    # demo_extracted = pc_kp_tc[8][0].float()
    B,N,_ = batch_pc.shape
    mu, sigma = probalistic_torch(batch_pc)
    # import pdb; pdb.set_trace()
    # sigma_v = np.linalg.inv(sigma)
    sigma_v = torch.inverse(sigma)
    # generate the 21 dim vector
    sigma_f = sigma.reshape(B,-1,9)
    sigma_vf = sigma_v.reshape(B,-1,9)

    # xyz = np.concatenate([mu, sigma_f, sigma_vf], axis = -1)
    xyz_tensor = torch.cat([mu, sigma_f, sigma_vf], dim = -1) #.to(device)
    # in tensor
    # xyz_tensor = torch.tensor(xyz, dtype=torch.float).to(device)
    # xyz_tensor = xyz_tensor[None,...]
    
    nsample_1 = 512
    PA_FPSTEST = PointnetSAModuleDebug(npoint=10+1,  # mainly for downsampling
                radius=15,  # to a larger one
                nsample=nsample_1,
                mlp=[21, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True)

    # nsample_1 = 128
    # PA_FPSTEST = PointnetSAModuleDebug(npoint=128+1,  # mainly for downsampling
    #             radius=2,
    #             nsample=nsample_1,
    #             mlp=[21, 64, 64, 128],
    #             use_xyz=True,
    #             normalize_xyz=True)
                
    # nsample_2 = 64
    # PA_FPSTEST2 = PointnetSAModuleDebug(npoint=64+1 ,
    #             radius=10,
    #             nsample=nsample_2,
    #             mlp=[21, 64, 64, 128],
    #             use_xyz=True,
    #             normalize_xyz=True)

    # nsample_3 = 32
    # PA_FPSTEST3 = PointnetSAModuleDebug(npoint=32+1 ,
    #             radius=100,
    #             nsample=nsample_3,
    #             mlp=[21, 64, 64, 128],
    #             use_xyz=True,
    #             normalize_xyz=True)

    # nsample_4 = 16
    # PA_FPSTEST4 = PointnetSAModuleDebug(npoint=8+1 ,
    #             radius=600,
    #             nsample=nsample_4,
    #             mlp=[21, 64, 64, 128],
    #             use_xyz=True,
    #             normalize_xyz=True)


    # group 1 
    cat_xyz = get_model_corners(xyz_tensor)
    inds, group_inds, debug_xyz1 = PA_FPSTEST(cat_xyz)
    mask = ((group_inds[:,:,0:1].repeat(1,1,nsample_1) - group_inds) !=0)
    mask[:,:,0] = True
    group_mu, group_sigma, xyz2 = group_probalistic(cat_xyz, group_inds, mask) # First point is omitted
    flag = torch.sum(mask,dim=-1)

    # # group 2 
    # cat_xyz2 = get_model_corners(xyz2)
    # inds2, group_inds2, debug_xyz2 = PA_FPSTEST2(cat_xyz2)
    # mask2 = ((group_inds2[:,:,0:1].repeat(1,1,nsample_2) - group_inds2) !=0)
    # mask2[:,:,0] = True
    # group_mu2, group_sigma2, xyz3  = group_probalistic(cat_xyz2, group_inds2, mask2) # First point is omitted
    # flag2 = torch.sum(mask2,dim=-1)
    # # print(flag2)   

    # cat_xyz3 = get_model_corners(xyz3)
    # inds3, group_inds3, _ = PA_FPSTEST3(cat_xyz3)
    # mask3 = ((group_inds3[:,:,0:1].repeat(1,1,nsample_3) - group_inds3) !=0)
    # mask3[:,:,0] = True
    # group_mu3, group_sigma3, xyz4  = group_probalistic(cat_xyz3, group_inds3, mask3) # First point is omitted
    # flag3 = torch.sum(mask3,dim=-1)
    # # print(flag3)

    # cat_xyz4 = get_model_corners(xyz4)
    # inds4, group_inds4, _ = PA_FPSTEST4(cat_xyz4)
    # mask4 = ((group_inds4[:,:,0:1].repeat(1,1,nsample_4) - group_inds4) !=0)
    # mask4[:,:,0] = True
    # group_mu4, group_sigma4, xyz5  = group_probalistic(cat_xyz4, group_inds4, mask4) # First point is omitted
    # flag4 = torch.sum(mask4,dim=-1)

    # remove the false positive
    # print(flag4)
    # NMS for points

    # threshold = 10 # TODO: hyperparameters  It cannot be added, delete the positive point
    group_mu = group_mu[:,1:,:]
    final_det = torch.log(1 / torch.det(group_sigma))
    final_det = final_det[:,1:]
    sort_index = torch.argsort(final_det, dim=-1,descending=True)
    sort_index = sort_index.unsqueeze(-1).repeat(1,1,3)
    sorted_group_mu = torch.gather(group_mu, dim=1, index = sort_index)

    # det_mask = (final_det > threshold)
    
    nms_mask = p_nms(sorted_group_mu) # delete the init one [:,1:,:]
    select_mask =  nms_mask  #* det_mask
    # group_mu = group_mu[:,1:,:]
    # final_pred = group_mu4[0,select_mask[0,:],:]
    return sorted_group_mu, select_mask, group_inds # return the group index for grouping

def pc_fused_connection(batch_pc_k, indx_match, select_2d_match, viewidx_match, shift_size, cam_info, pose_2d_collect, pose_2d_related, device,root_idx = 0, max_people = 10):
    # batch pc is the list
    # process the root_idx first
    ## do not set the batch_pc_k with batch size, batch_pc_k is the list
    # loop in the batch dim
    valid_num = (shift_size * 2) ** 2 + 1
    batch_pc_b = batch_pc_k[root_idx]
    batch_num = len(batch_pc_b)
    batch_pred_3d = torch.zeros(batch_num, max_people,cfg.DATASET.KEYPOINT.NUM, 4)
    for batch_idx, batch_pc in enumerate(batch_pc_b):
        if len(batch_pc) == 0: # root has no points
            continue
        batch_pc = batch_pc.float() # keep the psedo batch dim
        # batch_center = select_2d_match[root_idx].float()
        B,N,_ = batch_pc.shape
        mu, sigma = probalistic_torch(batch_pc)
        sigma_v = torch.inverse(sigma)
        # generate the 21 dim vector
        sigma_f = sigma.reshape(B,-1,9)
        sigma_vf = sigma_v.reshape(B,-1,9)

        # xyz = np.concatenate([mu, sigma_f, sigma_vf], axis = -1)
        xyz_tensor = torch.cat([mu, sigma_f, sigma_vf], dim = -1) #.to(device)
        # in tensor
        
        nsample_1 = 512
        npoints = 10 # 10
        PA_FPSTEST = PointnetSAModuleDebug(npoint=npoints+1,  # mainly for downsampling
                    radius=13,  # to a larger one 15
                    nsample=nsample_1,
                    mlp=[21, 64, 64, 128],
                    use_xyz=True,
                    normalize_xyz=True)

        # group 1 
        cat_xyz = get_model_corners(xyz_tensor)
        inds, group_inds, debug_xyz1 = PA_FPSTEST(cat_xyz) # B X N X n_sample
        # add the further cooresponding check module
        group_idx_root = group_inds[:,1:,:].clone() - 1  # delete the first  and assume others do not use the 0 (center point) # B x 10 X N_SAMPLE
        # select the center point and filter
        dim_n = 7 # 6 + 1
        # orig_xyz_cp = batch_center[:,None,:,:].repeat(1,npoints,1,1)
        # group_inds_tools = group_idx_root[:,:,:,None].repeat(1,1,1,dim_n)
        # extracted_center = torch.gather(orig_xyz_cp,dim=2,index=group_inds_tools.type(torch.int64)) # B x npoints x s_samples x dim_n
        

        mask = ((group_inds[:,:,0:1].repeat(1,1,nsample_1) - group_inds) !=0)
        center_filter = (group_inds != 0)
        mask[:,:,0] = True
        mask = mask * center_filter
        # final_mask = mask[:,1:,:]
        group_mu, group_sigma, xyz2, vali_mask  = group_probalistic(cat_xyz, group_inds, mask) # First point is omitted
        flag = torch.sum(mask,dim=-1)
        # using flag label to delete the false positive
        limit_mask = (flag[:,1:] > valid_num)
        # threshold = 10 # TODO: hyperparameters  It cannot be added, delete the positive point  # we need to set the threshold
        group_mu = group_mu[:,1:,:]
        final_det = torch.log(1 / torch.det(group_sigma))
        final_det = final_det[:,1:]
        sort_index = torch.argsort(final_det, dim=-1,descending=True)

        limit_mask = torch.gather(limit_mask, dim=1, index = sort_index)
        sort_index_base = sort_index.unsqueeze(-1).repeat(1,1,3) # the general sort index
        # sort_index_general = sort_index.unsqueeze(-1).repeat(1,1,4)
        # sort_index_center = sort_index[...,None,None].repeat(1,1,nsample_1,dim_n)
        sort_index_mask = sort_index[...,None].repeat(1,1,nsample_1)

        sorted_group_mu_root = torch.gather(group_mu, dim=1, index = sort_index_base) # after sort
        # sorted_fmask = torch.gather(final_mask, dim=1, index = sort_index_mask) # for updating the mask and 
        # sorted_mcenter = torch.gather(extracted_center, dim=1, index = sort_index_center)
        # sorted_group_idx = torch.gather(group_idx_root, dim=1, index=sort_index_mask)
        nms_mask = p_nms(sorted_group_mu_root) # delete the init one [:,1:,:]
        select_mask =  nms_mask * limit_mask # TODO: do not use the limit mask  
        # get the root joint related
        
        ###################  vis
        # inds = inds.squeeze(0)
        # group_inds = group_inds.squeeze(0)
        # extracted_xyz = cat_xyz[0,inds[4].long(),:] #.cpu().numpy()
        # debug_extracted_xyz = cat_xyz[0, group_inds[4].cpu().long(),:] #.cpu().numpy()
        # extract_pose_mu = sorted_group_mu_root[0,select_mask[0,...],:]
        # # test the KL distance
        # extracted_xyz = extracted_xyz.unsqueeze(0).unsqueeze(0)
        # extracted_xyz = extracted_xyz.repeat(1,512,1)
        # debug_extracted_xyz = debug_extracted_xyz.unsqueeze(0)
        # debug_dis = BKL_Div(extracted_xyz[0,:,:3], debug_extracted_xyz[0,:,:3], extracted_xyz[0,:,3:12], debug_extracted_xyz[0,:,3:12])
        # fig = plt.figure(figsize=(20, 15))
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(mu[0,:,0].cpu().numpy(), mu[0,:,1].cpu().numpy(), mu[0,:,2].cpu().numpy(), marker='o', s = 4)
        # # ax.scatter(extracted_xyz[:,0,0].cpu().numpy(), extracted_xyz[:,0,1].cpu().numpy(), extracted_xyz[:,0,2].cpu().numpy(), marker='^',s = 640)
        # ax.scatter(extract_pose_mu[:,0].cpu().numpy(), extract_pose_mu[:,1].cpu().numpy(), extract_pose_mu[:,2].cpu().numpy(), marker='^',s = 640)
        # # debug one point
        # ax.scatter(debug_extracted_xyz[0,:,0].cpu().numpy(),debug_extracted_xyz[0,:,1].cpu().numpy(),debug_extracted_xyz[0,:,2].cpu().numpy(),marker='o', s = 16)
        # # ax.scatter(extracted_xyz[:,0], extracted_xyz[:,1], extracted_xyz[:,2], marker='^',s = 640)
        # plt.savefig('debugfps.png')
        # import pdb; pdb.set_trace()
        ####################
        # TODO: for one batch
        # sorted_mcenter = sorted_mcenter[select_mask]
        # sorted_fmask = sorted_fmask[select_mask]
        # sorted_group_idx = sorted_group_idx[select_mask]


        # pred_num = sorted_mcenter.shape[0]
        # unique_mask = unique(sorted_mcenter)
        
        # # for one batch
        # for p in range(pred_num): # pred_num is the group num
        #     p_center = sorted_mcenter[p][unique_mask[p]] 
        #     contri_views_num = p_center.shape[0]
        #     # operate only exists enough views information
        #     if contri_views_num >= 3:
        #         mu_c, sigma_c = probalistic_torch(p_center[None,...])
        #         prepared_list = list(range(contri_views_num))
        #         comb_index = torch.tensor(list(itertools.combinations(prepared_list, 2)))
        #         mu_comp0 = mu_c[0,comb_index[:,0],...]
        #         mu_comp1 = mu_c[0,comb_index[:,1],...]
        #         sigma_comp0 = sigma_c[0,comb_index[:,0],...]
        #         sigma_comp1 = sigma_c[0,comb_index[:,1],...]
        #         KL_distance = BKL_Div(mu_comp0,mu_comp1,sigma_comp0,sigma_comp1)
        #         Ds = torch.zeros(contri_views_num,contri_views_num).to(device)
        #         # import pdb; pdb.set_trace()
        #         Ds[comb_index[:,0],comb_index[:,1]] = KL_distance
        #         Ds = Ds + Ds.T
        #         # Dis_sum = torch.sum(Ds, axis=-1)
        #         threshold = 13 # 13 is the hyper parameter
        #         remove_outliers = Ds > threshold
        #         remove_outliers_m = torch.sum(remove_outliers, dim=-1)
        #         remove_outliers_m = remove_outliers_m > (contri_views_num/2)
        #         invalid_center = p_center[remove_outliers_m,...]
        #     # Using the new information and get the new mask
        #     else:
        #         continue

            
        #     if len(invalid_center) > 0:
        #         tool_dis = sorted_mcenter[p][:,None,:].repeat(1,len(invalid_center),1)
        #         update_mask = torch.any(torch.norm(tool_dis - invalid_center, dim=-1) < 0.01 , dim=-1)
        #         update_mask = ~update_mask
        #         sorted_fmask[p] = sorted_fmask[p] * update_mask
        #         group_mu, _, _, _  = group_probalistic(cat_xyz[:,1:,:], sorted_group_idx[None,p:p+1,:], sorted_fmask[None,p:p+1,:])

            

        # select_mask, sorted_mcenter to optimize the result


        # sorted_group_mu_root = sorted_group_mu_root
        # select_mask = select_mask # general for all the joints

        for b in range(B): # just one batch first B == 1 # b == 0
            if torch.sum(select_mask[b,...]) == 0:
                continue # gap this batch_idx
            extract_pose = sorted_group_mu_root[b,...]
            root_pose = extract_pose[select_mask[b,...],:]
            group_idx_root_b = group_idx_root[b,...] # 10 X 512
            pred_nposes = root_pose.shape[0] # set the max people number
            pred_pose = torch.ones((pred_nposes,cfg.DATASET.KEYPOINT.NUM, 4)).to(device)   
            pred_pose[:,root_idx,:3] = root_pose # The root also need to select the matched 2d 
            # need to anti the false positive 
            # for k in range(NUM_LIMBS):
            for k in range(cfg.DATASET.KEYPOINT.NUM): #cfg.DATASET.KEYPOINT.NUM
                # src_k = joint_to_limb_heatmap_relationship[k][0] # NO USE
                # rethink the root joint

                # if k == root_idx:  # TODO: this changed
                #     continue
                dst_k = k
                # generate the input
                # judge if it has the points
                if len(batch_pc_k[dst_k][batch_idx]) == 0:
                    # it has no info here
                    pred_pose[:,dst_k,:] = torch.zeros(pred_nposes,4)
                    continue
                per_batch_pc = batch_pc_k[dst_k][batch_idx][b,...].float()
                per_batch_center = select_2d_match[dst_k][batch_idx][b,...].float()
                orig_xyz_cp = per_batch_center[None,:,:].repeat(npoints,1,1)
                per_batch_view = viewidx_match[dst_k][batch_idx][b:b+1,...].repeat(npoints, 1) # 采样
                mu, sigma = probalistic_torch(per_batch_pc.unsqueeze(0))
                sigma_v = torch.inverse(sigma)
                # generate the 21 dim vector
                sigma_f = sigma.reshape(1,-1,9)
                sigma_vf = sigma_v.reshape(1,-1,9)

                # xyz = np.concatenate([mu, sigma_f, sigma_vf], axis = -1)
                xyz_tensor = torch.cat([mu, sigma_f, sigma_vf], dim = -1)


                match_idx = indx_match[dst_k][batch_idx][b] # the same as B x 10 x 512
                new_group_idx = match_idx[group_idx_root_b.to(torch.int64)]
                # recovered the psedo batch dim , do not consider the outliers
                # new_group_idx = new_group_idx[None,...]
                mask1 = ((new_group_idx[:,0:1].repeat(1,nsample_1) - new_group_idx) !=0)
                mask1[:,0] = True
                mask2 = (new_group_idx != -1) # 
                mask = mask1 * mask2 # mask to ensure not sample the -1 index
                # new_group_idx = new_group_idx.cpu()
                new_group_idx[new_group_idx==-1] = 0  # All is -1
                # new_group_idx = new_group_idx.to(device)

                # refined new_group_idx
                sorted_new_group_idx = torch.gather(new_group_idx, dim=0, index = sort_index_mask[b,...])
                sorted_group_indx_tool = sorted_new_group_idx[:,:,None].repeat(1,1,dim_n)
                extracted_center = torch.gather(orig_xyz_cp,dim=1,index=sorted_group_indx_tool.type(torch.int64))
                sorted_fmask = torch.gather(mask, dim=0, index = sort_index_mask[b,...])
                extracted_view = torch.gather(per_batch_view, dim=1, index=sorted_new_group_idx.type(torch.int64))

                # import pdb; pdb.set_trace()
                sorted_mcenter = extracted_center[select_mask[b,...]] # select mask is select the group, just for batch = 1
                sorted_fmask = sorted_fmask[select_mask[b,...]]
                sorted_new_group_idx = sorted_new_group_idx[select_mask[b,...]]
                sorted_view = extracted_view[select_mask[b,...]]
                # check the all false 

                if xyz_tensor.numel() == 0 or sorted_new_group_idx.numel() == 0 or sorted_fmask.numel() == 0:
                    pred_pose[:,dst_k,:] = torch.zeros(pred_nposes,4)
                    continue

                group_mu, _, _, vali_mask  = group_probalistic(xyz_tensor, sorted_new_group_idx[None,...], sorted_fmask[None,...]) 
                group_mu = torch.cat([group_mu,vali_mask], dim=-1)
                pred_num = sorted_mcenter.shape[0]
                unique_mask = unique(sorted_mcenter)

                ## update the mu according to the 2D mismatch
                for p in range(pred_num):
                    p_center = sorted_mcenter[p][unique_mask[p]] 
                    p_views = sorted_view[p][unique_mask[p]]
                    # generate the 2D rela
                    for center, view in zip(p_center, p_views):
                        pose_2d_related[view.item()][batch_idx, p, k, :] = center

                    # # delete the mismatch process
                    # invalid_center = [] # initialize the invalid center
                    # # import pdb; pdb.set_trace()  index using example cam_g = cam_info[p_views[0].item()]
                    # contri_views_num = p_center.shape[0]
                    # # operate only exists enough views information
                    # if contri_views_num >= 3:
                    #     # varify the outliers by line distance
                    #     prepared_list = list(range(contri_views_num))
                    #     comb_index = torch.tensor(list(itertools.combinations(prepared_list, 2)))
                    #     pos_comp0 = p_center[comb_index[:,0],:3]
                    #     pos_comp1 = p_center[comb_index[:,1],:3]
                    #     di_comp0 = p_center[comb_index[:,0],3:6]
                    #     di_comp1 = p_center[comb_index[:,1],3:6]
                    #     pos_con = pos_comp1 - pos_comp0
                    #     norm_con = torch.cross(di_comp0, di_comp1)
                    #     norm_con = norm_con / torch.norm(norm_con, dim=-1, keepdim=True)
                    #     line_dis = torch.abs(torch.sum(pos_con *  norm_con, dim=-1))
                        
                    #     Ds = torch.zeros(contri_views_num,contri_views_num).to(device)
                    #     Ds[comb_index[:,0],comb_index[:,1]] = line_dis
                    #     Ds = Ds + Ds.T
                    #     threshold = 3  # 3 is 2.10
                    #     remove_outliers = Ds > threshold
                    #     remove_outliers_m = torch.sum(remove_outliers, dim=-1)
                    #     remove_outliers_m = remove_outliers_m > (contri_views_num/2)
                    #     invalid_center = p_center[remove_outliers_m,...]
                    #     invalid_views = p_views[remove_outliers_m]

                    #     # # Dis_sum = torch.sum(Ds, axis=-1)
                    #     # threshold = 13 # 13 is the hyper parameter
                    #     # remove_outliers = Ds > threshold
                    #     # remove_outliers_m = torch.sum(remove_outliers, dim=-1)
                    #     # remove_outliers_m = remove_outliers_m > (contri_views_num/2)
                    #     # invalid_center = p_center[remove_outliers_m,...]
                    # # Using the new information and get the new mask
                    # # else:
                    #     # group the mu
                    #     # continue

                    # if len(invalid_center) > 0:
                    #     for e_view, e_center in zip(invalid_views, invalid_center):
                    #         cam_g = cam_info[int(e_view.item())]
                    #         extracted_bakp = pose_2d_collect[batch_idx][int(e_view.item())][:,k,:] # N x 3  # change the original pred
                    #         e_2d = projectjointsPoints_torch(e_center[None,:3], cam_g['K'],cam_g['R'],cam_g['t'], cam_g['distCoef']) #TODO: It can simplified
                    #         p_idx = torch.argmin(torch.norm(extracted_bakp[:,:2] - e_2d , dim=-1))
                    #         extracted_bakp[p_idx,2] = 0.1 # to low level
                    #     if len(invalid_center) == contri_views_num:
                    #         group_mu[:,p:p+1,3] = 0.4
                    #         continue
                    #     tool_dis = sorted_mcenter[p][:,None,:].repeat(1,len(invalid_center),1)
                    #     update_mask = torch.any(torch.norm(tool_dis - invalid_center, dim=-1) < 0.01 , dim=-1)
                    #     update_mask = ~update_mask
                    #     sorted_fmask[p] = sorted_fmask[p] * update_mask
                    #     if torch.sum(sorted_fmask[p]) == 0:
                    #         # do not changed the value, and just give the mask label
                    #         group_mu[:,p:p+1,3] = 0.4
                    #         continue
                    #     update_mu, _, _, update_mask  = group_probalistic(xyz_tensor, sorted_new_group_idx[None,p:p+1,:], sorted_fmask[None,p:p+1,:])
                    #     update_mu = torch.cat([update_mu, update_mask], dim=-1)
                    #     group_mu[:,p:p+1,...] = update_mu



                ## update end



                # group_mu, _, _, vali_mask  = group_probalistic(xyz_tensor, new_group_idx, mask)
                # group_mu = torch.cat([group_mu,vali_mask], dim=-1)
                # sorted_c_mu = torch.gather(group_mu, dim=1, index = sort_index_general)
                # # extract the pose
                # sorted_c_mu = sorted_c_mu.cpu()
                # extract_pose_c = sorted_c_mu[b,...]
                # c_pose = extract_pose_c[select_mask[b,...],:] # use the select_mask

                
                pred_pose[:,dst_k,:] = group_mu
                # pred_pose[:,dst_k,:] = c_pose
            # valid_f = (pred_pose[:,:,3] != 0)
            # valid_p = torch.all(valid_f, dim=-1)
            # pred_pose = pred_pose[valid_p,:,:3]
            batch_pred_3d[batch_idx,:pred_nposes,:,:] = pred_pose
    
    # process for the nan value, can not be nan : double guarantee
    nan_judge = torch.any(torch.isnan(batch_pred_3d), dim=-1, keepdim=True)
    batch_pred_3d[...,3:4] = batch_pred_3d[...,3:4] * (~nan_judge)
    batch_pred_3d[torch.isnan(batch_pred_3d)] = 0  #
    # 0 tag has 0 value
    zero_judge = (batch_pred_3d[...,3:4] != 0) 
    batch_pred_3d[...,:3] = batch_pred_3d[...,:3] * (zero_judge)
    
    return batch_pred_3d #pred_pose # TODO: for one batch # return contains the valid mask

if __name__ == '__main__':
    # main()
    main2()


## debug code
# inds = inds.squeeze(0)
# group_inds = group_inds.squeeze(0)
# extracted_xyz = xyz_tensor[inds[4].long(),:]#.cpu().numpy()
# debug_extracted_xyz = xyz_tensor[group_inds[4].long(),:]#.cpu().numpy()
# # test the KL distance
# extracted_xyz = extracted_xyz.unsqueeze(0).unsqueeze(0)
# extracted_xyz = extracted_xyz.repeat(1,64,1)
# debug_extracted_xyz = debug_extracted_xyz.unsqueeze(0)
# debug_dis = BKL_Div(extracted_xyz, debug_extracted_xyz)
# print(debug_dis)
# # import pdb; pdb.set_trace()
# fig = plt.figure(figsize=(20, 15))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], marker='o', s = 4)
# ax.scatter(extracted_xyz[:,0,0].cpu().numpy(), extracted_xyz[:,0,1].cpu().numpy(), extracted_xyz[:,0,2].cpu().numpy(), marker='^',s = 640)
# # debug one point
# ax.scatter(debug_extracted_xyz[0,:,0].cpu().numpy(),debug_extracted_xyz[0,:,1].cpu().numpy(),debug_extracted_xyz[0,:,2].cpu().numpy(),marker='o', s = 16)
# # ax.scatter(extracted_xyz[:,0], extracted_xyz[:,1], extracted_xyz[:,2], marker='^',s = 640)
# plt.savefig('debugfps.png')

# import pdb; pdb.set_trace()