import numpy as np
import cvxpy as cp
import torch

def back_projection(x, d, K, Kd):  # changed into the kd 
    """
    Back project 2D points x(2xN) to the camera coordinate and ignore distortion parameters.
    :param x: 2*N
    :param d: real depth of every point
    :param K: camera intrinsics
    :param Kd: camera distorsion paras
    :return: X (Nx3), points in 3D
    """
    X = np.zeros((len(d), 3), np.float)
    # X[:, 0] = (x[:, 0] - K[0, 2]) * d / K[0, 0]
    # X[:, 1] = (x[:, 1] - K[1, 2]) * d / K[1, 1]
    X[:, 0] = (x[:, 0] - K[0, 2]) / K[0, 0]
    X[:, 1] = (x[:, 1] - K[1, 2]) / K[1, 1]
    # process the distorsion parameters undistorsion
    
    K_diff = np.zeros(12)
    K_diff[:5] = Kd
    x_bak = X[:,0].copy()
    y_bak = X[:,1].copy()

    for _ in range(5):
        r2 = X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]
        icdist = (1 + ((K_diff[7] * r2 + K_diff[6])*r2 + K_diff[5])*r2) / (1 + ((K_diff[4]*r2 + K_diff[1])*r2 + K_diff[0])*r2)
        deltaX = 2 * K_diff[2] * X[:, 0] * X[:, 1] + K_diff[3]*(r2 + 2*X[:, 0] * X[:, 0])+  K_diff[8]*r2+K_diff[9]* r2 *r2
        deltaY = K_diff[2]*(r2 + 2*X[:, 1] *X[:, 1]) + 2*K_diff[3]*X[:, 0] * X[:, 1]+ K_diff[10] * r2 + K_diff[11]* r2 *r2

        X[:, 0] = (x_bak - deltaX) *icdist
        X[:, 1] = (y_bak - deltaY) *icdist

    X[:, 0] = X[:, 0] * d
    X[:, 1] = X[:, 1] * d
    X[:, 2] = d
    return X

def back_projection_torch(x, d, K, Kd, device):  # changed into the kd 
    """
    Back project 2D points x(2xN) to the camera coordinate and ignore distortion parameters.
    :param x: 2*N
    :param d: real depth of every point
    :param K: camera intrinsics
    :param Kd: camera distorsion paras
    :return: X (Nx3), points in 3D
    """
    X = torch.zeros((len(d), 3), dtype=torch.float).to(device)
    # X[:, 0] = (x[:, 0] - K[0, 2]) * d / K[0, 0]
    # X[:, 1] = (x[:, 1] - K[1, 2]) * d / K[1, 1]
    X[:, 0] = (x[:, 0] - K[0, 2]) / K[0, 0]
    X[:, 1] = (x[:, 1] - K[1, 2]) / K[1, 1]
    # process the distorsion parameters undistorsion
    
    K_diff = torch.zeros(12).to(device)
    K_diff[:5] = Kd
    x_bak = X[:,0].clone()
    y_bak = X[:,1].clone()

    for _ in range(5):
        r2 = X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]
        icdist = (1 + ((K_diff[7] * r2 + K_diff[6])*r2 + K_diff[5])*r2) / (1 + ((K_diff[4]*r2 + K_diff[1])*r2 + K_diff[0])*r2)
        deltaX = 2 * K_diff[2] * X[:, 0] * X[:, 1] + K_diff[3]*(r2 + 2*X[:, 0] * X[:, 0])+  K_diff[8]*r2+K_diff[9]* r2 *r2
        deltaY = K_diff[2]*(r2 + 2*X[:, 1] *X[:, 1]) + 2*K_diff[3]*X[:, 0] * X[:, 1]+ K_diff[10] * r2 + K_diff[11]* r2 *r2

        X[:, 0] = (x_bak - deltaX) *icdist
        X[:, 1] = (y_bak - deltaY) *icdist

    X[:, 0] = X[:, 0] * d
    X[:, 1] = X[:, 1] * d
    X[:, 2] = d
    return X

def back_to_global(x, d, K, Kd, R,t):
    X = np.zeros((x.shape[0], 3), np.float)
    # X[:, 0] = (x[:, 0] - K[0, 2]) * d / K[0, 0]
    # X[:, 1] = (x[:, 1] - K[1, 2]) * d / K[1, 1]
    
    X[:, 0] = (x[:, 0] - K[0, 2]) / K[0, 0]
    X[:, 1] = (x[:, 1] - K[1, 2]) / K[1, 1]
    # process the distorsion parameters undistorsion
    K_diff = np.zeros(12)
    K_diff[:5] = Kd
    x_bak = X[:,0].copy()
    y_bak = X[:,1].copy()

    for _ in range(5):
        r2 = X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]
        icdist = (1 + ((K_diff[7] * r2 + K_diff[6])*r2 + K_diff[5])*r2) / (1 + ((K_diff[4]*r2 + K_diff[1])*r2 + K_diff[0])*r2)
        deltaX = 2 * K_diff[2] * X[:, 0] * X[:, 1] + K_diff[3]*(r2 + 2*X[:, 0] * X[:, 0])+  K_diff[8]*r2+K_diff[9]* r2 *r2
        deltaY = K_diff[2]*(r2 + 2*X[:, 1] *X[:, 1]) + 2*K_diff[3]*X[:, 0] * X[:, 1]+ K_diff[10] * r2 + K_diff[11]* r2 *r2

        X[:, 0] = (x_bak - deltaX) *icdist
        X[:, 1] = (y_bak - deltaY) *icdist

    X[:, 0] = X[:, 0] * d
    X[:, 1] = X[:, 1] * d
    X[:, 2] = d
    # X is in local coordinate
    # using R,t to transform to global coordinate
    X_global = (np.linalg.pinv(R) @ (X.transpose() - t)).transpose()
    return X_global


def back_to_global_torch(x, K, Kd, R,t, device):
    X = torch.zeros((x.shape[0], 3), dtype = torch.float).to(device)
    K = torch.from_numpy(K).to(device)
    Kd = torch.from_numpy(Kd).to(device)
    R = torch.from_numpy(R).to(device)
    t = torch.from_numpy(t).to(device)
    # X[:, 0] = (x[:, 0] - K[0, 2]) * d / K[0, 0]
    # X[:, 1] = (x[:, 1] - K[1, 2]) * d / K[1, 1]
    X[:, 0] = (x[:, 0] - K[0, 2]) / K[0, 0]
    X[:, 1] = (x[:, 1] - K[1, 2]) / K[1, 1]
    # process the distorsion parameters undistorsion
    K_diff = torch.zeros(12).to(device)
    K_diff[:5] = Kd
    x_bak = X[:,0].clone()
    y_bak = X[:,1].clone()

    for _ in range(5):
        r2 = X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]
        icdist = (1 + ((K_diff[7] * r2 + K_diff[6])*r2 + K_diff[5])*r2) / (1 + ((K_diff[4]*r2 + K_diff[1])*r2 + K_diff[0])*r2)
        deltaX = 2 * K_diff[2] * X[:, 0] * X[:, 1] + K_diff[3]*(r2 + 2*X[:, 0] * X[:, 0])+  K_diff[8]*r2+K_diff[9]* r2 *r2
        deltaY = K_diff[2]*(r2 + 2*X[:, 1] *X[:, 1]) + 2*K_diff[3]*X[:, 0] * X[:, 1]+ K_diff[10] * r2 + K_diff[11]* r2 *r2

        X[:, 0] = (x_bak - deltaX) *icdist
        X[:, 1] = (y_bak - deltaY) *icdist

    X[:, 0] = X[:, 0] * x[:,2]
    X[:, 1] = X[:, 1] * x[:,2]
    X[:, 2] = x[:,2]
    # X is in local coordinate
    # using R,t to transform to global coordinate
    # X_global = (np.linalg.pinv(R) @ (X.transpose() - t)).transpose()
    X_global = (torch.inverse(R) @ (X.t() - t)).t()
    return X_global


def get_3d_points(pred_bodys, root_depth, K, Kd, root_n=2): # add the distorsion process
    bodys_3d = np.zeros(pred_bodys.shape, np.float)
    bodys_3d[:, :, 3] = pred_bodys[:, :, 3]
    for i in range(len(pred_bodys)):
        if pred_bodys[i][root_n][3] == 0:
            continue
        pred_bodys[i][:, 2] += root_depth[i] # direct add # become the absolute depth
        bodys_3d[i][:, :3] = back_projection(pred_bodys[i][:, :2], pred_bodys[i][:, 2], K, Kd)
    return bodys_3d

def get_3d_points_torch(pred_bodys, root_depth, K, Kd, device, root_n=2): # add the distorsion process
    bodys_3d = torch.zeros(pred_bodys.shape, dtype=torch.float).to(device) 
    bodys_3d[:, :, 3] = pred_bodys[:, :, 3]
    for i in range(len(pred_bodys)):
        if pred_bodys[i][root_n][3] == 0:
            continue
        pred_bodys[i][:, 2] += root_depth[i] # direct add
        bodys_3d[i][:, :3] = back_projection_torch(pred_bodys[i][:, :2], pred_bodys[i][:, 2], K, Kd, device)
    return bodys_3d


def projectjointsPoints(X, K, R, t, Kd):
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    # input is the N*3
    X = X.transpose()
    x = (np.dot(R, X) + t)  # panoptic to kinect color scaling  cm to m 

    x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    # 去畸变
    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                            r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                            r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    # depth_val_norm = depth_val * W / f  # absolute depth sensing
    pose_2d = x[:2,:].copy().transpose()

    return pose_2d

def projectjointsPoints_cp(X, K, R, t, Kd):
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    # input is the N*3
    X = X.T
    x = R@X + t  # panoptic to kinect color scaling  cm to m 

    # x[0:2, :] = x[0:2, :] / (x[2:3, :] + 1e-5)
    x = x[0:2, :] / (x[2:3, :] + 1e-5)

    # r = cp.multiply(x[0, :] , x[0, :]) + cp.multiply(x[1, :] , x[1, :])

    # 去畸变
    # x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
    #                     ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
    #                         r + 2 * x[0, :] * x[0, :])
    # x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
    #                     ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
    #                         r + 2 * x[1, :] * x[1, :])

    # for debug

    # x0 = cp.multiply(x[0, :] , (1 + cp.multiply(Kd[0] , r) + cp.multiply(Kd[1], cp.power(r,2)) + cp.multiply(Kd[4],cp.power(r,3))
    #                     )) + cp.multiply(2 * Kd[2], cp.multiply(x[0, :],x[1, :])) + cp.multiply(Kd[3] , (
    #                         r + cp.multiply(2 , cp.multiply(x[0, :] , x[0, :]))))
    
    
    # x1 = cp.multiply(x[1, :] , (1 + cp.multiply(Kd[0] , r) + cp.multiply(Kd[1], cp.power(r,2)) + cp.multiply(Kd[4],cp.power(r,3))
    #                     )) + cp.multiply(2 * Kd[3], cp.multiply(x[0, :],x[1, :])) + cp.multiply(Kd[2] , (
    #                         r + cp.multiply(2 , cp.multiply(x[1, :] , x[1, :]))))

    # x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    # x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]
    o_0 = cp.multiply(K[0, 0] , x[0,:]) + cp.multiply(K[0, 1] , x[1,:]) + K[0, 2]
    o_1 = cp.multiply(K[1, 0] , x[0,:]) + cp.multiply(K[1, 1] , x[1,:]) + K[1, 2]

    o = cp.vstack([o_0,o_1])
    # depth_val_norm = depth_val * W / f  # absolute depth sensing
    # pose_2d = x[:2,:].copy().T
    pose_2d = o.T

    return pose_2d


def projectjointsPoints_torch(X, K, R, t, Kd):
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    # input is the N*3
    X = X.T
    x = R@X+t  # panoptic to kinect color scaling  cm to m 

    # x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)
    new_x = x[0:2, :] / (x[2, :] + 1e-5)

    r = new_x[0, :] * new_x[0, :] + new_x[1, :] * new_x[1, :]

    # 去畸变
    x0 = new_x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[2] * new_x[0, :] * new_x[1, :] + Kd[3] * (
                            r + 2 * new_x[0, :] * new_x[0, :])
    x1 = new_x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[3] * new_x[0, :] * new_x[1, :] + Kd[2] * (
                            r + 2 * new_x[1, :] * new_x[1, :])

    o_0 = K[0, 0] * x0 + K[0, 1] * x1 + K[0, 2]
    o_1 = K[1, 0] * x0 + K[1, 1] * x1 + K[1, 2]

    # depth_val_norm = depth_val * W / f  # absolute depth sensing
    pose_2d_t = torch.cat([o_0[None,:], o_1[None,...]], dim=0)
    pose_2d = pose_2d_t.T

    return pose_2d