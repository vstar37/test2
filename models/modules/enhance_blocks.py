import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.decoder_blocks import BasicDecBlk


class RectifyPrototypes(nn.Module):
    def __init__(self, feat_dim):
        """
        初始化 RectifyPrototypes 类，包含原型修正的层。

        Args:
            feat_dim (int): 特征维度（例如原型的特征维度）
        """
        super(RectifyPrototypes, self).__init__()
        # 用于细化修正后的原型的全连接层和 LayerNorm
        self.fc = nn.Linear(feat_dim, feat_dim, bias=False)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, support_es, query_es, prototypes):
        """
        根据支持集与查询集显式特征，调整原型。

        Args:
            support_es: [B, C_support, N] 支持集显式特征
            query_es: [B, C_query, N] 查询集显式特征
            prototypes: 支持集原型列表，每个原型为 (feat_dim,)

        Returns:
            prototypes_rectified_list: 修正后的原型列表，每个原型为形状 (feat_dim,)
        """
        # 1. 计算通道的平均值
        support_channel_mean = support_es.mean(dim=-1)  # [B, C_support]
        query_channel_mean = query_es.mean(dim=-1)  # [B, C_query]

        # 2. 计算通道之间的相关性
        relation_matrix = torch.einsum('bc,bk->ck', support_channel_mean, query_channel_mean)  # [C_support, C_query]
        relation_matrix = relation_matrix / (relation_matrix.size(-1) ** 0.5)  # 归一化处理
        relation_matrix_softmax = F.softmax(relation_matrix, dim=1)  # [C_support, C_query]

        # 3. 用相关性调整原型
        aligned_prototypes = torch.matmul(prototypes, relation_matrix_softmax)  # [N_prototypes, C_query]


        # 5. 融合残差原型
        feature_memory = self.fc(aligned_prototypes) + prototypes  # 通过全连接层细化修正后的原型
        # feature_memory = self.layer_norm(output + prototypes)  # 加入残差，经过归一化

        # 6. 转换为列表
        # prototypes_rectified_list = torch.unbind(output, dim=0)  # 每个原型为形状 (feat_dim,)
        feature_memory = feature_memory / torch.norm(feature_memory, dim=-1, keepdim=True)
        # print('feature_memory.shape = {}'.format(feature_memory.shape))

        return feature_memory