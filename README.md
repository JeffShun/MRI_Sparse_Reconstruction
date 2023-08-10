## 模型介绍
端到端MRI稀疏重建网络模型

## 文件结构说明

### 训练文件目录

- train/train.py: 单卡训练代码入口
- train/train_multi_gpu.py: 分布式训练代码入口
- train/custom/dataset/dataset.py: dataset类
- train/custom/model/model_network.py: 模型网络
- train/custom/model/model_head.py: 模型head
- train/custom/model/model_loss.py 模型loss
- train/custom/model/backbones/*.py 模型backbone
- train/custom/utils/generate_dataset.py: 从原始数据生成输入到模型的数据，供custom/dataset/dataset.py使用
- train/custom/utils/save_torchscript.py: 生成模型对应的静态图
- train/custom/utils/convert_rt.py: pytorch模型转tensorrt
- train/custom/utils/common_tools.py: 工具函数包，提供损失函数，数据增强和一些其它工具函数
- train/custom/utils/distributed_utils.py: 分布式训练工具函数
- train/custom/utils/cal_matrics.py: 统计模型结果的准确率
- train/custom/utils/test_script.py: 循环遍历各epoch模型，打印评估指标
- train/config/reconstruction_config.py: 训练的配置文件

### 预测文件目录

* exmple/detect_keypoint.yaml: 预测配置文件
* exmple/main.py: 预测入口文件
* infer/predictor.py: 模型预测具体实现，包括加载模型和后处理

## demo调用方法

1. 准备训练原始数据
   * 在train文件夹下新建train_data/origin_data文件夹，将训练集和验证集分别放入train、val文件夹

2. 生成处理后的训练数据，在train_data/processed_data文件夹下
   * cd train
   * python custom/utils/generate_dataset.py

3. 开始训练
   * 分布式训练：CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env train_multi_gpu.py
   * 单卡训练：python train.py
   
4. 准备测试数据
   * 将预测数据放入example/data/input目录

5. 开始预测
   * cd example
   * python main.py

6. 结果评估
   * cd train/cd custom/utils/
   * python cal_matrics.py
