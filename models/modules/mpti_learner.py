""" MPTI with/without attention Learner for Few-shot 3D Point Cloud Semantic Segmentation

Author: Zhao Na, 2020
"""
import os
import torch
from torch import optim
from torch.nn import functional as F

from models.baseline import MyNetwork
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint
from utils.loss import computeCrossEntropyLoss
from config import Config


class MPTILearner(object):
    def __init__(self, args, mode='train'):
        self.config = Config()
        self.model = MyNetwork(args)
        if torch.cuda.is_available():
            self.model.cuda()

        if mode=='train':
            # for default settings
            self.optimizer = torch.optim.Adam(
                [
                 {'params': self.model.encoder.local_encoder.parameters(), 'lr': 0.0001},
                 {'params': self.model.encoder.es_encoder.parameters()},
                 {'params': self.model.encoder.base_learner.parameters()},
                 {'params': self.model.encoder.linear_mapper.parameters()},
                 {'params': self.model.decoder.parameters()},
                ], lr=args.lr)

            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size,
                                                          gamma=args.gamma)


    def train(self, data):
        
        [support_x, support_y, query_x, query_y, support_names, query_names, support_xyz_mins, query_xyz_mins] = data

        self.model.train()

        query_logits, loss = self.model(data)

        [bce, loss_p1, loss_p2] = loss
        #if batch_idx > 2000:
            # 超过2000 iterations 后使用固定权重
            # loss = 20 * bce + 15 * loss_p1 + 15 * loss_p2
            #loss = 30 * bce + 10 * loss_p1 + 10 * loss_p2
        #else:
        lambdas = self.config.lambdas_pix_last
        lambda_bce = lambdas.get('bce', 1.0)  # 默认值为 1.0
        lambda_loss_p1 = lambdas.get('loss_p1', 1.0)  # 默认值为 1.0
        lambda_loss_p2 = lambdas.get('loss_p2', 1.0)  # 默认值为 1.0
        loss = (
                lambda_bce * bce +
                lambda_loss_p1 * loss_p1 +
                lambda_loss_p2 * loss_p2
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        query_pred = F.softmax(query_logits, dim=-1).argmax(dim=-1)
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0] * query_y.shape[1])

        return loss, accuracy

    def test(self, data):
        """
        Args:
            data = (support_ptclouds, support_masks, query_ptclouds, query_labels, support_filenames, query_filenames,  support_xyzmins, query_xyzmins)
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [support_x, support_y, query_x, query_y, support_names, query_names, support_xyz_mins, query_xyz_mins] = data
        self.model.eval()

        with torch.no_grad():
            logits, _ = self.model(data)  # 测试推理，输入4个输出两个
            pred = F.softmax(logits, dim=-1).argmax(dim=-1)
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return pred, accuracy

