import os
import torch
import math


class Config:
    def __init__(self):

        # get abs_dir
        self.abs_file = os.path.abspath(__file__)
        self.abs_dir = self.abs_file[:self.abs_file.rfind('\\')] if os.name == 'nt' else self.abs_file[
                                                                                         :self.abs_file.rfind(r'/')]
        # ========= Manually Setup =========#
        self.vis_test = False
        self.rectify = True
        self.use_x3d_es = True
        self.use_global_feature = True
        self.rectify_threshold = 0.9
        self.confidence_ratio_threshold = 0.3 #0.9
        self.diff_threshold = 0.3
        self.logits_strategy = [0,1,2][0]
        self.reconstruct_strategy = [0,1,2][0]

        self.task = ['few-shot'][0]
        self.train_method = ['local', 'cloud'][0]
        self.num_points = [2048, 20480][0]
        self.in_dim = 2460
        self.out_dim = 128

        self.compute_flops = True

        self.dec_target_size = [8, 16, 31, 61]
        self.optimizer = ['Adam', 'AdamW'][0]
        self.lr_schedule = ['StepLR', 'MultiStepLR', 'ExponentialLR', 'LinearLR', 'CyclicLR', 'OneCycleLR',
                            'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'LambdaLR', 'SequentialLR',
                            'ChainedScheduler', 'ConstantLR', 'ReduceLROnPlateau'][3]
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'pepper', 'crop'][:4]
        self.train_size = [224, 256, 352, 512][3]
        self.att_blk = ['SpatialAttention', 'ChannelAttention', 'MixResAttention'][2]
        self.IoU_finetune_last_epochs = [0, -2, -5, -10, -20][0]     # choose 0 to skip
        self.NonPrior_finetune_last_epochs = [0, -2, -5, -10, -20][3]
        self.Prior_finetune_first_epochs = [0, 2, 5, 10, 20][4]     # choose 0 to skip


        # filter configs
        self.gus_ker_type = ['2d', '3d'][0]  # build 2d gussian kernal or 3d
        self.verbose_eval = False
        # self.train_notice = [False, True][1]


        # ========= Automatically Configs =========#
        # Train Configs
        self.resume = True
        self.batch_size = {
            'local': 4,
            'cloud': 32
        }[self.train_method]
        self.num_workers = 6  # will be decrease to min(it, batch_size) at the initialization of the data_loader
        self.lr = 1e-5 * math.sqrt(self.batch_size / 5)  # adapt the lr linearly
        self.lr_decay_epochs = [1e4]    # Set to negative N to decay the lr in the last N-th epoch.
        self.lr_decay_rate = 0.5
        self.lambdas_pix_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 30 * 1,
            'loss_p1': 10 * 1,
            'loss_p2': 10 * 1
        }

        self.lambdas_pix_multi = {
            # not 0 means opening this loss
            'weight_pred_m': 1.0 * 1,
            'weight_mid_pred1': 0.5 * 1,
            'weight_mid_pred2': 0.5 * 1,
        }

        # Data Configs
        self.dataset = ['S3DIS'][0]
        self.s3dis_blockdir = '/home/amos/PycharmProjects/3DFewShot_TEST/datasets/S3DIS/blocks_bs1_s1/data_2048'

        # others
        self.device = [0, 'cpu'][0 if torch.cuda.is_available() else 1]     # .to(0) == .to('cuda:0')

        self.batch_size_valid = 1
        self.rand_seed = 7


'''
test = Config()

print(test.lambdas_pix_last)

'''
