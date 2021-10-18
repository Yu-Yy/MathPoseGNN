export CUDA_VISIBLE_DEVICES="3"
export PROJECT_HOME='/home/panzhiyu/project/3d_pose/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
#  "/home/panzhiyu/project/3d_pose/SMAP/model_logs_0629/stage3_root2/best_model.pth" #off_the_shelf
# '/home/panzhiyu/project/3d_pose/SMAP/model_logs_0629/stage3_root2/best_model.pth''/home/panzhiyu/project/3d_pose/SMAP/model_logs_0816/stage3_root2/iter-last.pth'
python test_gnn.py -p  "/home/panzhiyu/project/3d_pose/SMAP/model_logs_0929/stage3_root2/iter-201600.pth" \
-t generate_result \
-d test \
--batch_size 4