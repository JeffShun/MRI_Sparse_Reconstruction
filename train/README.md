
## 关键点检测项目

### 1）分布式训练

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env train_multi_gpu.py

### 2) 单卡训练

python train.py
