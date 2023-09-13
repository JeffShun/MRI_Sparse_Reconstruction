import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.model.backbones.MC_DUnet import *
from custom.model.model_head import *
from custom.model.model_network import *
from custom.model.model_loss import *
from custom.utils.common_tools import *
from custom.dataset.dataset import *

class network_cfg:
    # img
    patch_size = (320, 320)

    # network
    network = Model_Network(
        backbone = MC_DUnet(
            model=ResUnet,
            model_config = {
                'in_ch': 30,
                'out_chans': 30,
                'num_chans': 64,
                'n_res_blocks': 2,
                'global_residual': True,
                },

            datalayer = DataGDLayer,
            datalayer_config = {
                'learnable': True,
                'lambda_init': 1.0
                },
            num_iter=5,
            shared_params=True 
            ),
        head = Model_Head(in_channels=30,num_class=1),
        apply_sync_batchnorm=False,
    )

    # loss function
    loss_func = LossCompose([
        SSIMLoss(win_size = 7, k1 = 0.01, k2 = 0.03),
        MSELoss()
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
    log_dir = work_dir + "/Logs/MC_DUnet"
    checkpoints_dir = work_dir + '/checkpoints/MC_DUnet'
    checkpoint_save_interval = 1
    total_epochs = 100
    load_from = work_dir + '/checkpoints/pretrain/28.pth'

    # others
    device = 'cuda'
    dist_backend = 'nccl'
    dist_url = 'env://'
