import torch
import torch.nn as nn
from torch_scatter import scatter_mean

class PointHop(nn.Module):
    def __init__(self, args=None) -> None:
        super().__init__()
        self.outchannel = 6 + 24 + 9

    def forward(self, group_xyz, new_xyz, group_idx=None):
        B, N, K, C = group_xyz.shape

        X = group_xyz
        X = X.view(B * N, K, C)

        std_xyz = torch.std(group_xyz, dim=2, keepdim=True).view(B * N, 3)
        center = new_xyz.view(B * N, 3)

        idx = (X[:, :, 0] > 0).float() * 4 + (X[:, :, 1] > 0).float() * 2 + (X[:, :, 2] > 0).float()

        current_features = torch.zeros(B * N, 8, 3).to(group_xyz.device)
        current_features = scatter_mean(X, idx.long(), dim=1, out=current_features).view(B * N, 24)

        u, s, v = torch.linalg.svd(X)

        a1 = s[:, 0]
        a2 = s[:, 1]
        a3 = s[:, 2]

        Linearity = (a1 - a2) / (a1 + 1e-10)
        Planarity = (a2 - a3) / (a1 + 1e-10)
        Scattering = a3 / (a1 + 1e-10)

        u1 = torch.sum(s * torch.abs(v[:, :, 0]), dim=-1, keepdim=True)

        u2 = torch.sum(s * torch.abs(v[:, :, 1]), dim=-1, keepdim=True)
        u3 = torch.sum(s * torch.abs(v[:, :, 2]), dim=-1, keepdim=True)

        direction = torch.cat([u1, u2, u3], dim=-1)
        norm = v[:, :, 0]
        features = torch.cat([std_xyz, center, current_features, Linearity.unsqueeze(-1), Planarity.unsqueeze(-1),
                              Scattering.unsqueeze(-1), direction, norm], dim=-1)

        features = features.view(B, N, -1)
        return features

class GetExplicitStructure(nn.Module):
    def __init__(self, k_neighbors=32, out_dim=128):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.pointhop = PointHop()  # 引入 PointHop 计算显式结构

    def forward(self, point_cloud):
        """
        输入:
        - point_cloud: 输入的点云数据，形状 [batch_size, num_points, 3]，包含坐标 (x, y, z)

        输出:
        - explicit_structure: 计算得到的显式结构，形状 [batch_size, num_points, feature_dim]
        """
        batch_size, num_points, _ = point_cloud.shape

        # 步骤1：为每个点计算其 KNN，相邻的32个点
        group_idx, new_xyz, group_xyz = self.get_knn_groups(point_cloud)

        # 步骤2：通过 PointHop 计算显式结构
        explicit_structure = self.pointhop(group_xyz, new_xyz, group_idx)

        return explicit_structure

    def get_knn_groups(self, point_cloud):
        batch_size, num_points, _ = point_cloud.shape

        # 计算 KNN
        group_idx, new_xyz = self.knn_search(point_cloud, k=self.k_neighbors)

        # 获取邻居点坐标
        group_xyz = torch.gather(
            point_cloud.unsqueeze(2).expand(-1, -1, self.k_neighbors, -1),  # 扩展点云形状
            dim=1,
            index=group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # 扩展索引形状
        )

        return group_idx, new_xyz, group_xyz

    def knn_search(self, point_cloud, k):
        batch_size, num_points, _ = point_cloud.shape

        # Step 1: 扩展点云为两两点的欧几里得距离矩阵
        point_cloud_flattened = point_cloud.reshape(batch_size * num_points, 3)
        distance_matrix = torch.cdist(point_cloud_flattened, point_cloud_flattened, p=2)  # 计算所有点对之间的欧几里得距离

        # Step 2: 对于每个点，选取距离最近的k个点
        _, group_idx = distance_matrix.topk(k, largest=False, dim=1)  # 获取距离最近的k个点的索引
        group_idx = group_idx.reshape(batch_size, num_points, k)  # 重塑为 [batch_size, num_points, k]

        # Step 3: 确保索引在有效范围内
        group_idx = torch.clamp(group_idx, max=num_points - 1)  # 防止索引超出边界

        # Step 4: 计算每个分组的中心点
        new_xyz = torch.gather(point_cloud, 1, group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, 3))  # 获取每个点对应的中心点

        return group_idx, new_xyz


def main():
    print("exist debug random setting in es_encoder.py!")
    # 创建一个虚拟点云数据 [batch_size, num_points, 3]
    batch_size = 2
    num_points = 2048

    point_cloud = torch.randn(batch_size, num_points, 3)

    # 创建并测试显式结构计算
    get_es_model = GetExplicitStructure(k_neighbors=32)
    explicit_structure = get_es_model(point_cloud)

    print(explicit_structure.shape)  # 输出显式结构的形状

# 测试代码
#if __name__ == "__main__":
#    main()