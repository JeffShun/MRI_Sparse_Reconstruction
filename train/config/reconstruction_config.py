import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.dataset.dataset import MyDataset
from custom.model.DenosingComponents.ResUNet import ResUnet
from custom.model.DenosingComponents.DUNet import DIDN
from custom.model.DenosingComponents.SimpleCNN import SimpleCNN
from custom.model.DCLayers import DataPMLayer, DataGDLayer
from custom.model.backbones.My_DCNet import My_DCNet
from custom.model.backbones.DCFree import DCFree
from custom.model.model_head import Model_Head
from custom.model.model_network import Model_Network
from custom.model.model_loss import *
from custom.utils.common_tools import *


class network_cfg:
    # img
    patch_size = (320, 320)
    # network
    network = Model_Network(
        backbone = DCFree(
            model = ResUnet,
            model_config = {
                'in_chans': 2,
                'out_chans': 2,
                'num_chans': 64,
                'n_res_blocks': 2,
                'global_residual': True,
                },
            ),
        # backbone = My_DCNet(
        #     model=DIDN,
        #     model_config = {
        #         'in_chans': 2,
        #         'out_chans': 2,
        #         'num_chans': 64,
        #         'n_res_blocks': 5,
        #         'global_residual': True,
        #         },
        #     datalayer = DataPMLayer,
        #     datalayer_config = {
        #         'learnable': True,
        #         'lambda_init': 0.05
        #         },
        #     num_iter=8,
        #     shared_params=True 
        #     ),
        head = Model_Head(),
        apply_sync_batchnorm=False,
    )

    # loss function
    loss_func = LossCompose([
        # LossSamplingWrapper(SSIMLoss(win_size = 7, k1 = 0.01, k2 = 0.03), sampling_p=0.5),
        # LossSamplingWrapper(MSELoss(), sampling_p=0.5)
        SSIMLoss(win_size = 7, k1 = 0.01, k2 = 0.03),
        ])

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = None
        )
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/val.txt",
        transforms = None
        )
    
    # dataloader
    batchsize = 2
    shuffle = True
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-3
    weight_decay = 5e-4

    # scheduler
    milestones = [50,80]
    gamma = 0.1
    warmup_factor = 0.1
    warmup_iters = 50
    warmup_method = "linear"
    last_epoch = -1

    # debug
    valid_interval = 1
    log_dir = work_dir + "/Logs/DCFree"
    checkpoints_dir = work_dir + '/checkpoints/DCFree'
    checkpoint_save_interval = 1
    total_epochs = 100
    load_from = work_dir + '/checkpoints/pretrain/100.pth'

    # others
    device = 'cuda'
    dist_backend = 'nccl'
    dist_url = 'env://'
