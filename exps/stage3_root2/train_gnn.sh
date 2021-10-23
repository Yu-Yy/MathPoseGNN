export CUDA_VISIBLE_DEVICES="5,6"
export PROJECT_HOME='/home/panzhiyu/project/3d_pose/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python config.py -log
python -m torch.distributed.launch --nproc_per_node=2 train_gnnpose.py #--continue '/home/panzhiyu/project/3d_pose/SMAP/model_logs_1020resv2/stage3_root2/best_model.pth'