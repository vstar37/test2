import torch

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sampling(points, num_samples):
    """
    实现最远点采样（Farthest Point Sampling, FPS）算法
    Args:
        points: 输入点云数据，形状为 [in_channels, num_points]
        num_samples: 采样点的数量
    Returns:
        sampled_points: 采样后的点云，形状为 [in_channels, num_samples]
    """
    # 计算点云的总点数
    num_points = points.shape[1]

    # 如果点云数量小于需要采样的数量，返回原点云
    if num_points <= num_samples:
        return points

    # 初始化一个选择的点的索引
    selected_indices = torch.zeros(num_samples, dtype=torch.long)

    # 初始化距离数组，表示每个点与已选择点的最小距离
    distances = torch.full((num_points,), float('inf'))

    # 随机选择一个初始点作为第一个采样点
    selected_indices[0] = torch.randint(0, num_points, (1,)).item()

    # 循环采样其余的点
    for i in range(1, num_samples):
        # 计算当前所有点到已选择点的最小距离
        selected_points = points[:, selected_indices[:i]]
        dist = torch.cdist(points.transpose(0, 1), selected_points.transpose(0, 1))
        min_dist = dist.min(dim=1).values  # 找到每个点到最近已选点的最小距离

        # 更新最小距离
        distances = torch.minimum(distances, min_dist)

        # 选择距离最远的点
        selected_indices[i] = torch.argmax(distances)

    # 返回采样的点
    return points[:, selected_indices]



def biased_lagrangian_interpolation_direction(xyz, rgb, direction, num_infer):
    """
    对目标方向执行方向性插值。

    Args:
        xyz: 目标点云的坐标 [num_points, 3]。
        rgb: 目标点云的颜色 [num_points, 3]。
        direction: 插值方向 'a' 到 'h'。
        num_infer: 插值点的数量。

    Returns:
        interpolated_points: 插值后的点云，形状为 [num_infer, 6]，包含坐标和颜色信息。
    """
    # 计算目标方向的最小和最大坐标（例如根据方向的几何坐标计算）
    direction_min, direction_max = compute_direction_min_max(xyz, direction)
    # 确定插值位置
    interpolation_points = torch.linspace(direction_min, direction_max, num_infer, device=xyz.device)

    # 计算目标方向的权重偏置（例如，根据方向的几何坐标计算方向权重）
    direction_weights = compute_direction_weights(xyz, rgb, direction)

    # 初始化插值结果
    interpolated_points = torch.zeros(num_infer, 6, device=xyz.device)

    # 进行插值：每个方向的插值点数量为 num_infer // 3
    for i in range(num_infer):
        # 对于每个目标点，在方向约束下进行插值
        weighted_xyz = compute_weighted_interpolation(xyz, i, direction_weights)
        weighted_rgb = compute_weighted_interpolation(rgb, i, direction_weights)

        # 合并坐标和颜色信息：最终的插值点形状为 [num_infer, 6]
        interpolated_points[i, :3] = weighted_xyz  # 前3个维度是坐标
        interpolated_points[i, 3:] = weighted_rgb  # 后3个维度是颜色

    return interpolated_points


def compute_weighted_interpolation(points, idx, weights):
    """
    基于方向性权重进行插值计算。

    Args:
        points: 当前点云的坐标或颜色，形状为 [num_points, 3]。
        idx: 当前点的索引。
        weights: 方向性权重。

    Returns:
        weighted_point: 加权后的插值点。
    """
    # 使用权重对当前点进行插值
    weighted_point = (weights[idx].view(1, -1) * points).sum(dim=0)  # 计算加权平均坐标
    return weighted_point


def compute_direction_weights(xyz, rgb, direction):
    """
    计算方向性偏置的权重，用于影响插值结果。

    Args:
        xyz: 目标点云的坐标 [num_points, 3]。
        direction: 插值方向 'a' 到 'h'。

    Returns:
        direction_weights: 形状为 [num_points] 的方向性权重。
    """
    # 根据方向和坐标计算权重（可以自定义权重计算方式）
    direction_map = {
        'a': 0.1, 'b': 0.2, 'c': 0.3, 'd': 0.4,
        'e': 0.5, 'f': 0.6, 'g': 0.7, 'h': 0.8
    }

    # 获取方向权重
    direction_weight = direction_map.get(direction, 1.0)  # 默认为1.0

    # 根据目标点坐标调整权重（可以根据需要添加更多的逻辑）
    direction_weights = torch.full((xyz.shape[0],), direction_weight, device=xyz.device)
    return direction_weights


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx