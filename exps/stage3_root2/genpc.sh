export CUDA_VISIBLE_DEVICES="7"
export PROJECT_HOME='/home/panzhiyu/project/3d_pose/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
#  "/home/panzhiyu/project/3d_pose/SMAP/model_logs_0629/stage3_root2/best_model.pth" #generate_result off_the_shelf
# '/home/panzhiyu/project/3d_pose/SMAP/model_logs_0629/stage3_root2/best_model.pth''/home/panzhiyu/project/3d_pose/SMAP/model_logs_0816/stage3_root2/iter-last.pth'
python pc_gen.py -p  "/home/panzhiyu/project/3d_pose/SMAP/model_logs_0629/stage3_root2/best_model.pth" \
-t generate_result \
-d test \
--batch_size 4