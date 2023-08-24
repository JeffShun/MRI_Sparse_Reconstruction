import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.model.backbones.DC_MWCNN import *
from custom.model.backbones.Unet import *
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
        backbone = DC_MWCNN(in_ch=30,channels=32,stages=3),
        head = Model_Head(in_channels=30, num_class=1),
        apply_sync_batchnorm=False,
    )

    # loss function
    loss_func = LossCompose([
        SSIMLoss(win_size = 7, k1 = 0.01, k2 = 0.03),
        # SmoothL1Loss()
        # MSELoss()
        ])

    # dataset
    train_dataset = MyDataset(
        sample_list_file = work_dir + "/train_data/processed_data/train.txt",
        mask_file = work_dir + "/train_data/processed_data/sampling_masks/mask.h5",
        transforms = TransformCompose([
            to_tensor()
            ])
        )
    valid_dataset = MyDataset(
        sample_list_file = work_dir + "/train_data/processed_data/val.txt",
        mask_file = work_dir + "/train_data/processed_data/sampling_masks/mask.h5",
        transforms = TransformCompose([
            to_tensor()
            ])
        )
    
    # dataloader
    batchsize = 8
    shuffle = True
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-2
    weight_decay = 5e-4

    # scheduler
    milestones = [50,80]
    gamma = 0.1
    warmup_factor = 0.1
    warmup_iters = 50
    warmup_method = "linear"
    last_epoch = -1

    # debug
    version = "v2_Unet_Wavelet_DC"
    valid_interval = 2
    log_dir = work_dir + "/Logs/" + version
    checkpoints_dir = work_dir + '/checkpoints/' + version
    checkpoint_save_interval = 2
    total_epochs = 200
    load_from = work_dir + '/checkpoints/pretrain/28.pth'

    # others
    device = 'cuda'
    dist_backend = 'nccl'
    dist_url = 'env://'
