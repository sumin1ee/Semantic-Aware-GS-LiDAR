export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --config configs/semantickitti_nvs.yaml \
    --show_log