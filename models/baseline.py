import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbones.dgcnn import DGCNN
from models.backbones.es_encoder import GetExplicitStructure
from models.modules.enhance_blocks import RectifyPrototypes
from models.modules.decoder_blocks import BasicDecBlk
from models.quest import QUEST
from utils.dataloaders.loader import augment_pointcloud
from utils.encoder_utils import *

from config import Config
config = Config()



class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()
        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x

class MyNetwork(nn.Module):
    def __init__(self, args):
        super(MyNetwork, self).__init__()

        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.n_subprototypes = args.n_subprototypes
        self.k_connect = args.k_connect
        self.sigma = args.sigma
        self.n_classes = self.n_way+1
        self.encoder = Encoder(args)
        if config.use_global_feature:
            self.decoder = Decoder(args, 2460, 128)
        else:
            self.decoder = Decoder(args, 192, 128)



    def forward_process(self, x):
        n_way, k_shot, in_channels, num_points = x.shape
        return x.view(n_way * k_shot, in_channels, num_points)

    def forward(self, x):
        support_x, support_y, query_x, query_y, support_name, query_name, support_xyz_min, query_xyz_min = x
        support_x = self.forward_process(support_x)

        # encode
        if config.use_x3d_es:
            support_features, support_es = self.encoder(support_x, support_name, support_xyz_min)
            support_features = support_features / support_features.norm(dim=1, keepdim=True)
            support_es = support_es / support_es.norm(dim=1, keepdim=True)

            query_features, query_es = self.encoder(query_x, query_name, query_xyz_min)
            query_features = query_features / query_features.norm(dim=1, keepdim=True)
            query_es = query_es / query_es.norm(dim=1, keepdim=True)
            logits, loss = self.decoder(support_features, support_es, support_y, query_features, query_es, query_y)

        else:
            support_features = self.encoder(support_x, support_name, support_xyz_min)
            support_features = support_features / support_features.norm(dim=1, keepdim=True)

            query_features= self.encoder(query_x, query_name, query_xyz_min)
            query_features = query_features / query_features.norm(dim=1, keepdim=True)
            logits, loss = self.decoder(support_features, support_features, support_y, query_features, support_features, query_y)

        # decode
        return logits, loss

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dataset = args.dataset
        if self.dataset  == 's3dis':
            self.blocks_path = '/home/amos/PycharmProjects/3DFewShot_TEST/datasets/S3DIS/blocks_bs1_s1/data'
        elif self.dataset  == 'scannet':
            self.blocks_path = '/home/amos/PycharmProjects/3DFewShot_TEST/datasets/ScanNet/blocks_bs1_s1/data'
        else:
            print('Encoder does NOT find block path.')

        self.config = Config()

        # 无参数编码器
        if config.use_global_feature:
            self.global_encoder = Encoder_Seg()
        self.es_encoder = GetExplicitStructure(32, config.out_dim)

        # 参数化编码器
        self.local_encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)
        self.pc_augm_config = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter,
                         'shift': args.pc_augm_shift,
                         'random_color': args.pc_augm_color,
                         }

    def forward(self, x, x_name, x_xyz_min):
        local_feat = self.getLocalFeature(x)# (B, 192, N)

        # ====== step1: encode global features =====  selected_features, selected_es
        if config.use_global_feature:
            global_feat, es = self.getGlobalFeature(x, x_name, x_xyz_min) # (B, 4536, N)
            features =  torch.cat((global_feat, local_feat),1)

        elif config.use_x3d_es:
            es = self.getGlobalFeature(x, x_name, x_xyz_min)  # (B, 4536, N)
            features = local_feat
        else:
            features = local_feat

        features = features / features.norm(dim=1, keepdim=True)
        # 可以尝试对ES 也 归一化一下
        if config.use_x3d_es:
            return features, es #(B, 4536 + 192, N) 可能有比例失衡的潜在风险
        elif config.use_global_feature:
            return features, global_feat #(B, 4536 + 192, N) 可能有比例失衡的潜在风险
        else:
            return features

    def load_point_cloud(self, block_path, pc_attribs='xyzrgbXYZ'):
        data = np.load(block_path)  # data.shape = (num, 7)
        N = data.shape[0]  # number of points in this scan
        sampled_point_inds = np.random.choice(np.arange(N), 2048, replace=(N < config.num_points))

        np.random.shuffle(sampled_point_inds)

        data =  data[sampled_point_inds]
        # 分离 xyz、rgb 和 labels
        xyz = data[:, 0:3]
        rgb = data[:, 3:6]
        labels = data[:, 6].astype(np.int32)

        xyz_min = np.amin(xyz, axis=0)

        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ / xyz_max  # 归一化到 [0, 1]

        XYZ = augment_pointcloud(XYZ, self.pc_augm_config)

        ptcloud = []
        if 'xyz' in pc_attribs: ptcloud.append(xyz)  
        if 'rgb' in pc_attribs: ptcloud.append(rgb)  
        if 'XYZ' in pc_attribs: ptcloud.append(XYZ)  

        ptcloud = np.concatenate(ptcloud, axis=1)
        ptcloud = np.transpose(ptcloud)  
        # ptcloud = np.expand_dims(ptcloud, axis=0)  

        pc_tensor = torch.tensor(ptcloud).float().to('cuda')

        return pc_tensor, xyz_min

    def reconstruct_global_pc(self, current_block_pc, current_block_xyzmin, scene_name, current_block_name):
        """
        构建包含当前块及其邻居的全局点云，并生成索引掩码以提取当前块的特征。

        Returns:
            global_pc: [1, in_channels, num_points*9] 的全局点云
            index_mask: 索引掩码，用于在全局编码特征中定位当前块的点
        """
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        # 当前块的行列索引

        row_col_info = current_block_name.split("_row")[-1]

        current_row = int(row_col_info.split("_col")[0])
        current_col = int(row_col_info.split("_col")[-1])


        concat_blocks = []
        xyz_min_list = []
        index_offset = 0
        index_mask = []

        for row_offset, col_offset in neighbors:
            # 计算邻近块的行列索引
            neighbor_row = current_row + row_offset
            neighbor_col = current_col + col_offset

            # 正则匹配符合条件的文件名
            if self.dataset == 's3dis':
                pattern = re.compile(rf"{scene_name}_block_\d+_row{neighbor_row}_col{neighbor_col}\.npy")
            elif self.dataset == 'scannet':
                pattern = re.compile(rf"{scene_name}_block_\d+_row{neighbor_row}_col{neighbor_col}\.npy")

            matched_files = [f for f in os.listdir(self.blocks_path) if pattern.match(f)]

            if matched_files:
                neighbor_name = matched_files[0]
                neighbor_path = os.path.join(self.blocks_path, neighbor_name)

                if neighbor_name == f"{current_block_name}.npy":
                    concat_blocks.append(current_block_pc)
                    xyz_min_list.append(current_block_xyzmin)
                    index_mask.extend(range(index_offset, index_offset + current_block_pc.size(-1)))

                else:
                    neighbor_block_pc, neighbor_xyz_min = self.load_point_cloud(neighbor_path, pc_attribs='xyzrgbXYZ')
                    concat_blocks.append(neighbor_block_pc)
                    xyz_min_list.append(neighbor_xyz_min)
            else:
                concat_blocks.append(torch.zeros((9, 2048), dtype=torch.float32).to('cuda'))
                xyz_min_list.append([0, 0, 0])

            index_offset += 2048

        # 拼接所有块
        global_pc = torch.cat(concat_blocks, dim=-1)

        # 计算归一化坐标
        xyz = global_pc[0:3, :]  # 选择第1到第3行（即xyz坐标）
        xyz_min = torch.amin(xyz, dim=1, keepdim=True)
        XYZ = (xyz - xyz_min) / torch.amax(xyz - xyz_min, dim=1, keepdim=True)
        global_pc[6:9, :] = XYZ  # 更新 global_pc 的最后三行为归一化坐标
        return global_pc, torch.tensor(index_mask, dtype=torch.long).to(global_pc.device)

    def generate_global_pc(self, current_block_pc, current_block_xyzmin):
        """
        使用扩散推理重建当前点云块及其周围的八个邻居点云块。
        对每个方向 (a-h)，推理当前块中的最邻近 grid，生成对应方向的目标点云。

        Args:
            current_block_pc: 当前块的点云, 形状为 [in_channels, num_points].
            current_block_xyzmin: 当前块的最小坐标 [x_min, y_min, z_min].

        Returns:
            global_pc: 包含当前块及邻居块的扩散推理点云.
            index_mask: 当前块点云的掩码.
        """
        # 当前块的九宫格划分
        grid = {
            1: 'top-left', 2: 'top', 3: 'top-right',
            4: 'left', 5: 'center', 6: 'right',
            7: 'bottom-left', 8: 'bottom', 9: 'bottom-right'
        }

        # 每个方向的最邻近grid编号
        direction_to_grid = {
            'a': [4, 1, 2],  # 左上块
            'b': [1, 2, 3],  # 上方块
            'c': [2, 3, 6],  # 右上块
            'd': [1, 4, 7],  # 左方块
            'e': [3, 6, 9],  # 右方块
            'f': [4, 7, 8],  # 左下块
            'g': [6, 9, 8],  # 右下块
            'h': [7, 8, 9],  # 下方块
        }

        # 保存扩散推理的结果
        global_pc = []
        index_mask = []

        # 1. 划分当前块为九宫格
        grid_points = self.divide_block_into_grid(current_block_pc)

        # 2. 针对每个方向 (a-h)，进行扩散推理
        for direction_idx, direction in enumerate(direction_to_grid):
            # 获取目标方向的邻居grid点云
            neighbor_blocks = [
                current_block_pc[:, grid_points[grid_idx]] for grid_idx in direction_to_grid[direction]
            ]

            # 将这些grid点云传入扩散推理,返回227个点 (test
            inferred_points = self.diffusion_inference_for_direction(
                neighbor_blocks, direction
            )

            # 将推理结果存储
            global_pc.append(inferred_points)
            index_mask.extend([0] * inferred_points.shape[1])  # 推理结果掩码为0

            # 在第五次 (中心块插入的位置) 将当前块点云插入到结果
            if direction_idx == 3:
                global_pc.append(current_block_pc)
                index_mask.extend([1] * current_block_pc.shape[1])  # 当前块点云掩码为1

        # 3. 合并所有推理结果
        global_pc = torch.cat(global_pc, dim=1)  # [in_channels, total_num_points]

        return global_pc, index_mask

    def divide_block_into_grid(self, current_block_pc):
        """
        将当前块划分为九宫格，返回每个区域的点云索引。

        Args:
            current_block_pc: 当前块的点云，形状为 [in_channels, num_points]。

        Returns:
            grid_points: 划分后的九宫格，每个区域包含点的索引。
        """
        grid_points = {}

        x_coords = current_block_pc[0, :]  # x 坐标
        y_coords = current_block_pc[1, :]  # y 坐标

        x_min, x_max = torch.min(x_coords), torch.max(x_coords)
        y_min, y_max = torch.min(y_coords), torch.max(y_coords)

        x_step = (x_max - x_min) / 3
        y_step = (y_max - y_min) / 3

        for i in range(9):
            x_low = x_min + (i % 3) * x_step
            x_high = x_min + ((i % 3) + 1) * x_step
            y_low = y_min + (i // 3) * y_step
            y_high = y_min + ((i // 3) + 1) * y_step

            region_indices = torch.where(
                (x_coords >= x_low) & (x_coords < x_high) &
                (y_coords >= y_low) & (y_coords < y_high)
            )[0]

            grid_points[i + 1] = region_indices

        return grid_points

    def diffusion_inference_for_direction(self, neighbor_blocks, direction, num_sample = 341):
        """
        对目标方向进行扩散推理，推理出目标块的点云。

        Args:
            current_block_pc: 当前块的点云, 形状为 [in_channels, num_points].
            current_block_xyzmin: 当前块的最小坐标 [x_min, y_min, z_min].
            direction: 目标方向（a-h）.
            neighbor_blocks: 最邻近的三个点云块点云, 每个形状为 [in_channels, num_points_in_block].

        Returns:
            inferred_points: 推理得到的目标点云, 形状为 [in_channels, 227].
        """
        combined_pc = torch.cat(neighbor_blocks, dim=1)  # [in_channels, total_neighbor_points]
        # FPS 如果点数少于341个，返回一个全0形状为 [in_channels, 227]的空tensor
        total_points = combined_pc.shape[1]

        if total_points >= num_sample:
            combined_pc = farthest_point_sampling(combined_pc, num_sample)
        else:
            inferred_points = torch.zeros(combined_pc.shape[0], num_sample//3, device=combined_pc.device)
            return inferred_points

        xyz = combined_pc[:3, :]  # 提取坐标 [3, num_infer]
        rgb = combined_pc[3:6, :]  # 提取颜色 [3, num_infer]

        group_idx_xyz, new_xyz, group_xyz = self.get_knn_groups(xyz)
        group_idx_rgb, new_rgb, group_rgb = self.get_knn_groups(rgb)
        features = self.compute_clusters(group_xyz, group_rgb)
        xyz = features[:3, :]  # 提取坐标 [3, num_infer]
        rgb = features[3:6, :]  # 提取颜色 [3, num_infer]
        inferred_points = biased_lagrangian_interpolation_direction(xyz, rgb, direction, num_sample//3)

        return inferred_points

    def get_knn_groups(self, point_cloud, k=16):
        """
        获取点云的 KNN 分组。

        Args:
            point_cloud: 形状为 [num_points, 3] 的点云。
            k: 邻居的数量，默认为 16。

        Returns:
            group_idx: 邻居点的索引，形状为 [num_points, k]。
            new_xyz: 每个分组的中心点，形状为 [num_points, 3]。
            group_xyz: 每个分组的邻居点，形状为 [num_points, k, 3]。
        """
        # 获取 KNN 结果
        group_idx, new_xyz = self.knn_search(point_cloud, k=k)

        # 获取邻居点坐标
        group_xyz = torch.gather(
            point_cloud.unsqueeze(1).expand(-1, k, -1),  # [num_points, k, 3]
            dim=0,
            index=group_idx.unsqueeze(-1).expand(-1, -1, 3)  #  [num_points, k, 3]
        )

        return group_idx, new_xyz, group_xyz

    def knn_search(self, point_cloud, k):
        """
        计算KNN并返回点云的k个邻居。

        Args:
            point_cloud: 形状为 [3, num_points] 的点云。
            k: 邻居的数量。

        Returns:
            group_idx: 形状为 [num_points, k] 的邻居索引。
            new_xyz: 邻域点的中心点坐标，形状为 [num_points, 3]。
        """
        point_cloud = point_cloud.T  # 转为 [num_points, 3]
        distance_matrix = torch.cdist(point_cloud, point_cloud, p=2)  # [num_points, num_points]
        _, group_idx = distance_matrix.topk(k, largest=False, dim=1)  # [num_points, k]
        neighbors = torch.gather(
            point_cloud.unsqueeze(1).expand(-1, k, -1),  # [num_points, k, 3]
            dim=0,
            index=group_idx.unsqueeze(-1).expand(-1, -1, 3)  # [num_points, k, 3]
        )
        new_xyz = neighbors.mean(dim=1)  # [num_points, 3]

        return group_idx, new_xyz

    def compute_clusters(self, group_xyz, group_rgb):
        """
        计算每个点邻域的均值特征，包括坐标和颜色。

        Args:
            group_xyz: 形状为 [num_points, k, 3] 的邻域坐标集合。
            group_rgb: 形状为 [num_points, k, 3] 的邻域颜色集合。

        Returns:
            features: 形状为 [num_points, 6] 的特征，包含每个点的邻域均值坐标和颜色。
        """
        # 计算坐标的均值
        mean_xyz = group_xyz.mean(dim=1)  # 形状为 [num_points, 3]

        # 计算颜色的均值
        mean_rgb = group_rgb.mean(dim=1)  # 形状为 [num_points, 3]

        # 将均值坐标和均值颜色拼接为一个特征向量
        features = torch.cat([mean_xyz, mean_rgb], dim=1)  # 形状为 [num_points, 6]

        return features

    def compute_geometric_properties(self, clusters):
        """
        对特征点计算几何属性。

        Args:
            clusters: 每个簇的点云信息，形状为 [num_clusters, k, 3].

        Returns:
            geometric_properties: 每个特征点的几何属性 (Linearity, Planarity, Scattering)，
                                  形状为 [num_clusters, 3].
        """
        geometric_properties = []

        for cluster_xyz in clusters:  # [k, 3] 每个邻域的点
            print(cluster_xyz.shape)
            # 计算协方差矩阵
            cluster_centered = cluster_xyz - cluster_xyz.mean(dim=0, keepdim=True)  # 中心化
            cov_matrix = cluster_centered.T @ cluster_centered / (cluster_xyz.shape[0] - 1)  # [3, 3]

            # 计算特征值
            eigvals = torch.linalg.eigvalsh(cov_matrix)  # 实数特征值，升序排列
            eigvals = eigvals.sort(descending=True)[0]  # [3]，降序排列

            # 几何属性计算
            linearity = (eigvals[0] - eigvals[1]) / eigvals[0]
            planarity = (eigvals[1] - eigvals[2]) / eigvals[0]
            scattering = eigvals[2] / eigvals[0]

            geometric_properties.append([linearity, planarity, scattering])

        return torch.tensor(geometric_properties, device=clusters.device)  # [num_clusters, 3]


    def getGlobalFeature(self, x, x_name, x_xyz_min):
        '''
        Args:
            x: 支持集或查询集点云块，形状为 [batch_size, in_channels, num_points]
            x_name: 点云块文件名列表
            x_xyz_min: 点云块的最小坐标，用于对齐

        Returns:
            features_global: 编码后的全局特征
            xyz_index_global: 提取出的与 x 对应的全局特征
        '''

        with torch.no_grad():
            global_pcs = []
            index_masks = []

            # Debug: 确保 x_name 是字符串
            if isinstance(x_name, bytes):
                x_name = x_name.decode('utf-8')

            expanded_x_name = []
            for item in x_name:
                # 确保 item 是字符串
                if isinstance(item, bytes):
                    item = item.decode('utf-8')  # 解码为字符串
                item_cleaned = re.sub(r"(support|query)_\d+_", "", item)
                item_cleaned = item_cleaned.replace("\n", " ")  # 去掉换行符
                item_cleaned = item_cleaned.strip("[]").replace("'", "")  # 去掉括号和单引号

                names = item_cleaned.split()
                expanded_x_name.extend(names)  # 展开到最终列表中

            for current_block_pc, block_name, current_block_xyzmin in zip(x, expanded_x_name, x_xyz_min):
                if isinstance(block_name, bytes):
                    block_name = block_name.decode('utf-8')
                scene_name = block_name.split('_block')[0]
                current_block_name = block_name

                if config.reconstruct_strategy==0:
                    global_pc, index_mask = self.reconstruct_global_pc(
                        current_block_pc, current_block_xyzmin, scene_name, current_block_name
                    )
                else :
                    global_pc, index_mask = self.generate_global_pc(
                        current_block_pc, current_block_xyzmin
                    )
                global_pcs.append(global_pc.squeeze(0))  # 去除第一个维度
                index_masks.append(index_mask)

            global_pcs = torch.stack(global_pcs, dim=0)
            global_pcs = global_pcs.permute(0, 2, 1)  # 转置维度 [B, C, N] -> [B, N, C]

            global_pcs_XYZ = global_pcs[:, :, 6:9]
            global_pcs_xyz = global_pcs[:, :, :3]
            global_pcs_rgb = global_pcs[:, :, 3:6]

            # 继续计算显式结构
            es_XYZ = self.es_encoder(global_pcs_XYZ).permute(0, 2, 1) # 现在 global_pcs 形状为 [B, N, 3]
            es_xyz = self.es_encoder(global_pcs_xyz).permute(0, 2, 1) # 现在 global_pcs 形状为 [B, N, 3]
            es_rgb = self.es_encoder(global_pcs_rgb).permute(0, 2, 1) # 现在 global_pcs 形状为 [B, N, 3]

            es = torch.cat((es_xyz,es_rgb,es_XYZ), dim=1)

            selected_es = [es[i, :, mask].unsqueeze(0) for i, mask in enumerate(index_masks)]
            selected_es = torch.cat(selected_es, dim=0)  # [batch_size, out_dim, num_points]

            # 根据 index_mask 提取所需的特征
            if config.use_global_feature:
                global_pcs = global_pcs.permute(0, 2, 1)  # 转置维度 [B, N, C] -> [B, C, N]

                features_global = self.global_encoder(global_pcs)

                selected_features = [features_global[i, :, mask].unsqueeze(0) for i, mask in enumerate(index_masks)]
                selected_features = torch.cat(selected_features, dim=0)  # [batch_size, out_dim, num_points]

                return selected_features, selected_es
            else:
                return selected_es

    def getLocalFeature(self, x):
        # DGCNN
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        feat_level1, feat_level2 = self.local_encoder(x) #edgeconv_outputs[0] 的形状为 (B, 64, N)，out 的形状为 (B, 256, N)。
        feat_level3 = self.base_learner(feat_level2) #out_dim = 64
        map_feat = self.linear_mapper(feat_level2) # out_dim = 64
        return torch.cat((feat_level1, map_feat, feat_level3), dim=1) # dim = 64*3

class Decoder(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_points = args.pc_npts

        self.diff_threshold = config.diff_threshold
        self.dist_method = 'cosine'

        self.bn_fc = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Conv1d(in_dim, 196, 1),
            nn.BatchNorm1d(196),
            nn.ReLU(),
            nn.Conv1d(196, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        if config.rectify:
            self.pre_embed_1 = nn.Sequential(
                nn.Linear((6 + 24 + 9)*3, out_dim),
                nn.LayerNorm(out_dim)
            )
            self.pre_embed_2 = nn.Sequential(
                nn.Linear((6 + 24 + 9)*3, out_dim),
                nn.LayerNorm(out_dim)
            )
            self.enhance_block = RectifyPrototypes(out_dim)
            # es denoise
            self.denoise_q=nn.Conv1d(out_dim,out_dim,kernel_size=1)
            self.denoise_v=nn.Conv1d(out_dim,out_dim,kernel_size=1)
        elif config.use_x3d_es:
            # `print("using_x3d_es")
            self.pre_embed_1 = nn.Sequential(
                nn.Linear((6 + 24 + 9)*3, out_dim),
                nn.LayerNorm(out_dim)
            )
            self.pre_embed_2 = nn.Sequential(
                nn.Linear((6 + 24 + 9)*3, out_dim),
                nn.LayerNorm(out_dim)
            )
            self.quest = QUEST()
        else:
            self.quest = QUEST()

    def forward(self, support_features, support_es, support_y, query_features, query_es, query_y):
        '''
        :param support_features: B C N
        :param support_y:
        :param query_features: B C N
        :param query_y:
        :return:
        '''
        # channel reduce
        support_feat = self.bn_fc(support_features)
        query_feat = self.bn_fc(query_features)

        support_feat = support_feat.permute(0, 2, 1)
        query_feat = query_feat.permute(0, 2, 1)
        # k_shot = query_feat.shape[0]
        # obtain prototype_1
        feature_memory_list, label_memory_list = [], []
        support_feat = support_feat.view(self.n_way, self.k_shot, self.n_points, -1)

        mask_bg = (support_y == 0)  # Background mask
        bg_features = support_feat[mask_bg]  # Extract background features

        '''
        mask_fg = ~mask_bg  # 取反，得到前景掩码
        support_bg_feature = bg_features
        support_fg_feature = support_feat[mask_fg]  # 提取前景特征
        q_mask_bg = (query_y == 0)  # Background mask
        q_mask_fg = ~q_mask_bg  # 取反，得到前景掩码
        query_bg_features = query_feat[q_mask_bg]  # Extract background features
        query_fg_features = query_feat[q_mask_fg]  # Extract background features

        # ✅ 存储特征到 pth 文件
        feature_dict = {
            "support_bg_features": support_bg_feature.clone().detach() if support_bg_feature.numel() > 0 else None,
            "support_fg_features": support_fg_feature.clone().detach() if support_fg_feature.numel() > 0 else None,
            "query_bg_features": query_bg_features.clone().detach() if query_bg_features.numel() > 0 else None,
            "query_fg_features": query_fg_features.clone().detach() if query_fg_features.numel() > 0 else None,
        }

        torch.save(feature_dict, "debug_features.pth")
        '''

        if bg_features.shape[0] < 1:  # Handle edge case for no background points
            bg_features = torch.ones(1, support_feat.shape[-1]).cuda() * 0.1
        else:
            bg_features = bg_features.mean(0).unsqueeze(0)  # Mean pooling for background prototype
        feature_memory_list.append(bg_features)
        label_memory_list.append(torch.tensor(0).unsqueeze(0))  # Label 0 for background


        for i in range(self.n_way):  # Loop over foreground classes
            mask_fg = (support_y[i] == 1)  # Foreground mask for class `i`
            fg_features = support_feat[i, mask_fg]  # Extract foreground features
            fg_features = fg_features.mean(0).unsqueeze(0)  # Mean pooling for foreground prototype
            feature_memory_list.append(fg_features)
            label_memory_list.append(torch.tensor(i + 1).unsqueeze(0))  # Label `i+1` for foreground

        feature_memory = torch.cat(feature_memory_list, dim=0)
        label_memory = torch.cat(label_memory_list, dim=0).cuda()
        label_memory = F.one_hot(label_memory, num_classes=self.n_way + 1)
        feature_memory = feature_memory / torch.norm(feature_memory, dim=-1, keepdim=True)
        feature_memory_1 = feature_memory.unsqueeze(0).repeat(self.n_way, 1, 1)

        # get logits_1
        sim = [query_feat[i] @ feature_memory_1[i].t() for i in range(self.n_way)]
        sim = torch.stack(sim, dim=0)
        logits_1 = sim @ label_memory.float()

        # obtain prototype_2
        if config.rectify:
            B, C, N = support_es.shape
            support_es = support_es.permute(0, 2, 1).contiguous()  # [batch_size, N_points, feature_dim]
            support_es = support_es.view(-1, support_es.shape[-1])  # [batch_size * N_points, feature_dim]
            query_es = query_es.permute(0, 2, 1).contiguous()  # [batch_size, N_points, feature_dim]
            query_es = query_es.view(-1, query_es.shape[-1])  # [batch_size * N_points, feature_dim]
            support_es = self.pre_embed_1(support_es)
            query_es = self.pre_embed_2(query_es)
            support_es = support_es.view(B, -1, N)  # Reshape back
            query_es = query_es.view(B, -1, N)  # Reshape back

            # 降噪处理
            # 生成查询和值向量
            support_q = self.denoise_q(support_es)  # [B, N_points, feat_dim]
            support_v = self.denoise_v(support_es)  # [B, N_points, feat_dim]

            query_q = self.denoise_q(query_es)  # [B, N_points, feat_dim]
            query_v = self.denoise_v(query_es)  # [B, N_points, feat_dim]

            # 计算注意力分布
            support_attn = torch.einsum('bnd,bmd->bnm', support_q, support_es)  # [B, N_points, N_points]
            query_attn = torch.einsum('bnd,bmd->bnm', query_q, query_es)  # [B, N_points, N_points]

            support_attn = F.softmax(support_attn, dim=-1)  # [B, N_points, N_points]
            query_attn = F.softmax(query_attn, dim=-1)  # [B, N_points, N_points]

            # 生成修正特征
            support_correction = torch.sum(support_attn.unsqueeze(-1) * support_v.unsqueeze(1),
                                           dim=2)  # [B, N_points, feat_dim]
            query_correction = torch.sum(query_attn.unsqueeze(-1) * query_v.unsqueeze(1),
                                         dim=2)  # [B, N_points, feat_dim]
            # 更新特征
            support_es = support_es + support_correction  # [B, N_points, feat_dim]
            query_es = query_es + query_correction  # [B, N_points, feat_dim]

            feature_memory_2 = self.enhance_block(support_es, query_es, feature_memory_1)
        else:
            # use another rectify method
            if config.use_x3d_es:
                support_es = support_es.permute(0, 2, 1).contiguous()  # [batch_size, N_points, feature_dim]
                support_es = support_es.view(-1, support_es.shape[-1])  # [batch_size * N_points, feature_dim]
                query_es = query_es.permute(0, 2, 1).contiguous()  # [batch_size, N_points, feature_dim]
                query_es = query_es.view(-1, query_es.shape[-1])  # [batch_size * N_points, feature_dim]
                support_es = self.pre_embed_1(support_es)

                query_es = self.pre_embed_2(query_es)
                support_es = support_es.view(self.n_way, self.k_shot, self.n_points, -1)  # Reshape back
                query_es = query_es.view(self.n_way * self.k_shot, self.n_points, -1)  # Reshape back

                feature_memory_2 = self.quest(query_es, support_es, feature_memory_1)
            else:
                feature_memory_2 = self.quest(query_feat, support_feat, feature_memory_1)

        # get logits_2

        sim_rectified = [query_feat[i] @ feature_memory_2[i].t() for i in range(self.n_way)]
        sim_rectified = torch.stack(sim_rectified, dim=0)
        logits_2 = sim_rectified @ label_memory.float()

        # 计算 prototype_quality_loss
        prototype_quality_loss_1 = self.sup_regulize_Loss(feature_memory_1, support_feat, support_y)
        prototype_quality_loss_2 = self.sup_regulize_Loss(feature_memory_2, support_feat, support_y)

        ICDL_1 = self.inter_class_difference_loss(feature_memory_1)
        ICDL_2 = self.inter_class_difference_loss(feature_memory_2)

        # 测试 原型对支持集的表示能力 与 原型类间区分度
        loss_p1 = prototype_quality_loss_1 + ICDL_1
        loss_p2 = prototype_quality_loss_2 + ICDL_2

        # 计算 logits_1 不确定点比例
        probs = F.softmax(logits_1, dim=-1)  # 转化为概率分布
        top2_vals, _ = probs.topk(2, dim=-1)
        confidence_scores_1 = top2_vals[..., 0] - top2_vals[..., 1]  # 最大与次大值的差值

        low_confidence_mask = confidence_scores_1 < config.rectify_threshold  # 定义低置信度点
        low_confidence_ratio = low_confidence_mask.sum().item() / (low_confidence_mask.numel() + 1e-6)

        w_1 = 1 / (loss_p1 + 1e-6)
        w_2 = 1 / (loss_p2 + 1e-6)
        ac_1 = w_1 / (w_1 + w_2)
        ac_2 = w_2 / (w_1 + w_2)

        '''
        log_file = "low_confidence_ratio_log.txt"  # 记录文件路径

        def log_low_confidence_ratio(ratio_1, ratio_final):
            """即时写入每个 batch 的低置信度点占比"""
            with open(log_file, "a", buffering=1) as f:  # 行缓冲模式，每行写完立即刷新
                f.write(f"{ratio_1:.6f},{ratio_final:.6f}\n")  # 记录 batch_idx，方便绘图
                f.flush()  # 强制刷新缓冲区，确保写入
                os.fsync(f.fileno())  # 进一步确保写入磁盘（可选）
        '''
        # 决策逻辑
        if config.logits_strategy == 0:
            # 引入低置信度比率，如果原型1结果比率高，混合；比率低，不混合。
            if low_confidence_ratio < config.confidence_ratio_threshold:
                final_logits = logits_1
                print('strategy==0, p1 is good.')
            else:
                final_logits = ac_1 * logits_1 + ac_2 * logits_2

            # 计算 final_logits 低置信度比率
            probs_final = F.softmax(final_logits, dim=-1)
            top2_vals_final, _ = probs_final.topk(2, dim=-1)
            confidence_scores_final = top2_vals_final[..., 0] - top2_vals_final[..., 1]
            low_confidence_mask_final = confidence_scores_final < config.rectify_threshold
            #low_confidence_ratio_final = low_confidence_mask_final.sum().item() / (
            #            low_confidence_mask_final.numel() + 1e-6)

        elif config.logits_strategy == 1:
            # 高置信度点,直接使用 logits_1. 低置信度点混合
            final_logits = logits_1.clone()  # 初始化最终 logits
            final_logits[low_confidence_mask] = (
                    ac_1 * logits_1[low_confidence_mask] + ac_2 * logits_2[low_confidence_mask]
            )
            '''
            # 统计高置信度点和低置信度点的数量
            num_low_confidence = low_confidence_mask.sum().item()  # 低置信度点数量
            num_high_confidence = low_confidence_mask.numel() - num_low_confidence  # 高置信度点数量

            # 打印结果
            print(f"Number of high-confidence points: {num_high_confidence}")
            print(f"Number of low-confidence points: {num_low_confidence}")
            '''
        else:
            # 全部混合
            final_logits = ac_1 * logits_1 + ac_2 * logits_2

        # 计算对抗性平衡系数 alpha, 谁损失高 就增加权重，降低它
        alpha = 1 / (1 + torch.exp(-0.3 * (loss_p1 - loss_p2)))

        # 计算最终损失
        cross_entropy_loss = F.cross_entropy(final_logits.reshape(-1, self.n_way + 1), query_y.reshape(-1).long())
        loss_p1_weighted = (1 - alpha) * loss_p1
        loss_p2_weighted = alpha * loss_p2

        loss = [cross_entropy_loss, loss_p1_weighted, loss_p2_weighted]
        '''
        loss_log_file = "loss_log.txt"  # 记录文件路径

        def log_loss(cross_entropy_loss, loss_p1_weighted, loss_p2_weighted):
            """记录归一化后的损失日志，使其总和为1"""
            total_loss = cross_entropy_loss + loss_p1_weighted + loss_p2_weighted

            # 避免除零错误
            if total_loss > 1e-6:
                cross_entropy_loss /= total_loss
                loss_p1_weighted /= total_loss
                loss_p2_weighted /= total_loss

            with open(loss_log_file, "a", buffering=1) as f:  # 行缓冲模式，确保即时写入
                f.write(f"{cross_entropy_loss:.6f}, {loss_p1_weighted:.6f}, {loss_p2_weighted:.6f}\n")
                f.flush()  # 强制刷新缓冲区
                os.fsync(f.fileno())  # 确保写入磁盘（可选）

        # 记录低置信度点占比
        if config.logits_strategy == 0:
            log_low_confidence_ratio(low_confidence_ratio, low_confidence_ratio_final)
            log_loss(cross_entropy_loss.item(), loss_p1_weighted.item(), loss_p2_weighted.item())
        '''
        return final_logits, loss

    def sup_regulize_Loss(self, prototype_supp, supp_fts, support_y):
        """
        Compute the loss for prototype support alignment branch.

        Args:
            prototype_supp (torch.Tensor): Prototype embeddings for each way, shape [n_way, n_prototypes, feat_dim].
            supp_fts (torch.Tensor): Support features, shape [n_way, k_shot, num_points, feat_dim].
            support_y (torch.Tensor): Labels for support points, shape [n_way, k_shot, num_points].
        """
        n_ways, k_shots, num_points, feat_dim = supp_fts.shape
        _, n_prototypes, _ = prototype_supp.shape  # Extract number of prototypes (e.g., background, foreground).

        # Initialize total loss
        loss = 0

        # Iterate over each way (class)
        for way in range(n_ways):
            # Extract prototypes for this class (background + foreground)
            prototypes = prototype_supp[way]  # Shape: [n_prototypes, feat_dim]

            # Iterate over each shot
            for shot in range(k_shots):
                img_fts = supp_fts[way, shot]  # Shape: [num_points, feat_dim]
                point_labels = support_y[way, shot]  # Shape: [num_points]

                # Compute distances to prototypes
                supp_dist = [self.calculateSimilarity(img_fts, prototypes[i], self.dist_method) for i in
                             range(n_prototypes)]
                supp_pred = torch.stack(supp_dist, dim=1).squeeze(-1)  # Shape: [num_points, n_prototypes]

                # Create ground truth labels for foreground and background
                supp_label = torch.full_like(point_labels, 255,
                                             device=img_fts.device).long()  # Initialize to ignore index
                supp_label[point_labels == 1] = 1  # Foreground
                supp_label[point_labels == 0] = 0  # Background

                # Compute cross-entropy loss, ignoring points with label 255
                loss = loss + F.cross_entropy(supp_pred, supp_label, ignore_index=255) / k_shots / n_ways

        return loss
    def calculateSimilarity(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (num_points, feat_dim)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (num_points, 1)
        """
        if method == 'cosine':
            # Expand prototype to match feat's dimensions for cosine similarity
            prototype = prototype.unsqueeze(0)  # Shape: (1, feat_dim)
            similarity = F.cosine_similarity(feat, prototype, dim=1).unsqueeze(1) * scaler

        elif method == 'euclidean':
            # Expand prototype to match feat's dimensions for pairwise distance
            prototype = prototype.unsqueeze(0)  # Shape: (1, feat_dim)
            similarity = - F.pairwise_distance(feat, prototype, p=2) ** 2
            similarity = similarity.unsqueeze(1)
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity
    def channel_correlation_loss(self, prototypes, support_features, query_features):
        """
        通道关联性损失：减少原型与支持集特征和查询集特征之间的通道差异
        Args:
            prototypes: (num_classes, feature_dim) 类别原型
            support_features: (batch_size, num_channels, num_points) 支持集特征
            query_features: (batch_size, num_channels, num_points) 查询集特征

        Returns:
            loss: A scalar representing the channel correlation loss
        """
        # 支持集和查询集都需要通过通道与原型的关联来进行计算
        batch_size, num_channels, num_points = support_features.shape

        # 通过计算支持集和查询集特征的通道关联度来比较它们与原型的关系
        # 1. 支持集特征和原型之间的余弦相似度
        support_features = support_features.flatten(start_dim=1)  # (batch_size, num_channels * num_points)
        corr_support = F.cosine_similarity(support_features, prototypes.unsqueeze(0).expand(batch_size, -1), dim=1)  # (batch_size, num_classes)

        # 2. 查询集特征和原型之间的余弦相似度
        query_features = query_features.flatten(start_dim=1)  # (batch_size, num_channels * num_points)
        corr_query = F.cosine_similarity(query_features, prototypes.unsqueeze(0).expand(batch_size, -1), dim=1)  # (batch_size, num_classes)

        # 我们希望支持集和查询集特征与对应的原型相似度高
        # 计算支持集和查询集的平均相似度损失
        loss_support = torch.mean(torch.abs(corr_support - 1))  # 目标是与原型的相关性接近1
        loss_query = torch.mean(torch.abs(corr_query - 1))  # 目标是与原型的相关性接近1

        # 总损失是支持集和查询集损失的加权和
        loss = (loss_support + loss_query) / 2
        return loss
    def inter_class_difference_loss(self, prototypes):
        """
        类间差异损失：增大不同类别原型之间的距离，增加类别区分度
        Args:
            prototypes: A tensor of shape (batch_size, class_num, feature_dim), 表示所有类别的原型

        Returns:
            loss: A scalar representing the inter-class difference loss
        """
        batch_size, class_num, feature_dim = prototypes.shape

        # 计算原型之间的欧式距离（两两之间）
        prototypes_flattened = prototypes.view(batch_size, class_num, -1)  # (batch_size, class_num, feature_dim)
        dist_matrix = torch.cdist(prototypes_flattened, prototypes_flattened,
                                  p=2)  # shape: (batch_size, class_num, class_num)

        # 转化为差异性分值（1 - similarity），距离越大，差异性越小
        similarity_matrix = torch.exp(-dist_matrix)  # 可以使用负指数函数将距离转化为相似度
        difference_matrix = 1 - similarity_matrix  # 转化为差异性

        # 计算损失：对于差异性小于阈值的类别对，增加差异性
        loss = torch.zeros(batch_size, device=prototypes.device)

        # 只计算差异性小于阈值的部分
        for i in range(class_num):
            for j in range(i + 1, class_num):  # 避免对称计算
                # 如果差异性低于阈值，则计算损失并推动差异性增大
                # print(difference_matrix[:, i, j])
                mask = difference_matrix[:, i, j] < self.diff_threshold  # 只处理差异性小于阈值的类别对
                if mask.any():  # 如果存在需要计算损失的类别对
                    # 计算差异性和阈值的差距，并根据该差距计算损失
                    loss += torch.sum(torch.exp(self.diff_threshold - difference_matrix[:, i, j]) * mask)  # 非线性损失

        return loss.mean()  # 返回平均损失





class Args:
    def __init__(self):
        # 数据相关
        self.phase = 'graphtrain'  # 阶段选择：pretrain, finetune, prototrain, protoeval, mptitrain, mptieval
        self.dataset = 's3dis'  # 数据集名称：s3dis 或 scannet
        self.cvfold = 0  # 留一测试的折数
        self.data_path = './datasets/S3DIS/blocks_bs1_s1'  # 数据路径
        self.pretrain_checkpoint_path = None  # 预训练模型的路径
        self.model_checkpoint_path = None  # 模型训练的路径
        self.save_path = './log_s3dis/'  # 保存日志和检查点的目录
        self.eval_interval = 1500  # 每隔多少次迭代评估模型

        # 优化相关
        self.batch_size = 4  # 每个batch的任务数
        self.n_workers = 12  # 数据加载的线程数
        self.n_iters = 30000  # 训练的迭代次数
        self.lr = 0.001  # 学习率
        self.step_size = 5000  # 学习率衰减步长
        self.gamma = 0.5  # 学习率衰减系数

        # 预训练相关
        self.pretrain_lr = 0.001  # 预训练学习率
        self.pretrain_weight_decay = 0.0  # 权重衰减
        self.pretrain_step_size = 50  # 预训练学习率衰减步长
        self.pretrain_gamma = 0.5  # 预训练学习率衰减系数

        # Few-shot设置
        self.n_way = 2  # 支持集类别数
        self.k_shot = 1  # 支持集每类样本数
        self.n_queries = 1  # 查询集每类样本数
        self.n_episode_test = 100  # 测试时每次配置的任务数

        # 点云处理
        self.pc_npts = 2048  # 点云输入点数
        self.pc_attribs = 'xyzrgbXYZ'  # 点云特征：xyz（坐标），rgb（颜色），XYZ（归一化坐标）
        self.pc_augm = False  # 是否启用点云数据增强
        self.pc_augm_scale = 0.0  # 数据增强：缩放范围
        self.pc_augm_rot = 1  # 数据增强：绕z轴旋转
        self.pc_augm_mirror_prob = 0.0  # 数据增强：镜像概率
        self.pc_augm_jitter = 1  # 数据增强：高斯抖动
        self.pc_augm_shift=0.0
        self.pc_augm_color = 0


        # 特征提取网络配置
        self.dgcnn_k = 10  # DGCNN邻居数
        self.edgeconv_widths = [[64, 64], [64, 64], [64, 64]]  # EdgeConv层宽度
        self.dgcnn_mlp_widths = [512, 256]  # DGCNN中的MLP宽度
        self.base_widths = [128, 64]  # 基学习器的宽度
        self.output_dim = 64  # 输出特征的维度
        self.use_attention = False  # 是否使用注意力机制

        # ProtoNet配置
        self.dist_method = 'euclidean'  # 距离度量方法：cosine或euclidean

        # MPTI配置
        self.n_subprototypes = 100  # 每类支持集的子原型数
        self.k_connect = 200  # 最近邻用于构建局部约束亲和矩阵的数量
        self.sigma = 1.0  # 高斯相似度函数中的超参数

        # 自动解析
        self.pc_in_dim = len(self.pc_attribs)  # 输入点云的维度由属性长度决定

    def parse_edgeconv_widths(self):
        """返回解析后的EdgeConv宽度配置"""
        return self.edgeconv_widths

    def parse_dgcnn_mlp_widths(self):
        """返回解析后的DGCNN MLP宽度配置"""
        return self.dgcnn_mlp_widths

    def parse_base_widths(self):
        """返回解析后的BaseLearner宽度配置"""
        return self.base_widths

from fvcore.nn import FlopCountAnalysis

def main():
    print("exist debug random setting in baseline.py!")
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MyNetwork(args).to(device)

    # ✅ 构造随机的输入数据
    n_way = args.n_way
    k_shot = args.k_shot
    batch_size = n_way * k_shot
    num_points = 2048
    in_channels = 9

    support_x = torch.randn(n_way, k_shot, in_channels, num_points).to(device)
    support_y = torch.randint(0, 2, (n_way, k_shot, num_points)).to(device)
    query_x = torch.randn(batch_size, in_channels, num_points).to(device)
    query_y = torch.randint(0, 2, (batch_size, num_points)).to(device)

    def normalize_data(x):
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        eps = 1e-8
        return (x - x_min) / (x_max - x_min + eps)

    support_x = normalize_data(support_x)
    query_x = normalize_data(query_x)

    support_name = ["support_1_['Area_1_office_31_block_19_row1_col4']", "support_7_['Area_2_office_8_block_12_row2_col2']"]
    query_name = ["query_1_['Area_5_office_33_block_14_row3_col2']", "query_7_['Area_1_office_30_block_9_row1_col2']"]
    support_xyz_min = torch.randn(batch_size, 3).to(device)
    query_xyz_min = torch.randn(batch_size, 3).to(device)

    # ✅ 统一数据格式，确保是 tuple
    input_data = (
        support_x, support_y, query_x, query_y,
        support_name, query_name, support_xyz_min, query_xyz_min
    )

    # ✅ 计算 FLOPs
    flops = FlopCountAnalysis(model, (input_data,))
    total_flops = flops.total() / 1e9  # 转换为 GFLOPs
    print(f"模型 FLOPs: {total_flops:.2f} GFLOPs")

    # ✅ 进行推理
    pred, loss = model(input_data)
    print(f"Prediction shape: {pred.shape}")
    print(f"Loss: {loss}")

if __name__ == "__main__":
    main()
