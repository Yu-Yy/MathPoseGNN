export CUDA_VISIBLE_DEVICES="2,3"
export PROJECT_HOME='/home/panzhiyu/project/3d_pose/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python config.py -log
python -m torch.distributed.launch --nproc_per_node=2 train_gnn.py