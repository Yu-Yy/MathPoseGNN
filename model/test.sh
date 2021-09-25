export CUDA_VISIBLE_DEVICES="7"
export PROJECT_HOME='/home/panzhiyu/project/3d_pose/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME

python gnn_refinenet.py