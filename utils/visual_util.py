import os
import numpy as np
import open3d as o3d
import h5py
import glob
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import pickle



def visualize_point_cloud_with_dynamic_clipping(npy_file_path, target_class_id=None):
    """
    动态裁剪点云并可视化。
    用户可以通过设置 x、y 或 z 轴的裁剪范围来显示内部结构，
    并可设置指定类别点的颜色为黄色（接近木板的黄色），
    其他类别的点为黑色。

    参数:
    - npy_file_path (str): 点云数据的 .npy 文件路径。
    - target_class_id (int, optional): 指定要设置黄色的类别编号。
    """
    # 加载点云数据
    point_cloud_data = np.load(npy_file_path)


    # 随机采样点云，保留 sample_fraction 的点
    total_points = point_cloud_data.shape[0]
    sample_size = int(total_points * 1/2)
    print("total points: ", total_points)
    sampled_indices = np.random.choice(total_points, sample_size, replace=False)
    sampled_point_cloud_data = point_cloud_data[sampled_indices]
    point_cloud_data = sampled_point_cloud_data

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

    # 如果点云数据有颜色信息，则加载颜色
    if point_cloud_data.shape[1] >= 6:
        colors = point_cloud_data[:, 3:6] / 255.0  # 归一化颜色值到 [0, 1]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(point_cloud_data[:, :3]))

    # 获取点云边界
    min_bound = np.min(point_cloud_data[:, :3], axis=0)
    max_bound = np.max(point_cloud_data[:, :3], axis=0)

    # 初始化裁剪范围

    clipping_bounds = {
        'x_min': min_bound[0], 'x_max': max_bound[0],
        'y_min': min_bound[1], 'y_max': max_bound[1],
        'z_min': min_bound[2], 'z_max': max_bound[2],
    }

    # 定义裁剪函数
    def crop_point_cloud():
        """
        按照当前裁剪范围裁剪点云。
        """
        points = np.asarray(pcd.points)
        mask = (
                (points[:, 0] >= clipping_bounds['x_min']) & (points[:, 0] <= clipping_bounds['x_max']) &
                (points[:, 1] >= clipping_bounds['y_min']) & (points[:, 1] <= clipping_bounds['y_max']) &
                (points[:, 2] >= clipping_bounds['z_min']) & (points[:, 2] <= clipping_bounds['z_max'])
        )

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(points[mask])

        # 如果点云数据有颜色信息，则加载颜色
        if point_cloud_data.shape[1] >= 6:
            cropped_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

        return cropped_pcd

    # 可视化回调函数
    def update_vis(vis):
        cropped_pcd = crop_point_cloud()
        vis.clear_geometries()
        vis.add_geometry(cropped_pcd)

    # 定义快捷键功能
    def adjust_clipping(vis, axis, adjustment):
        """
        根据键盘调整裁剪范围。
        """
        if axis == 'x_min':
            clipping_bounds['x_min'] += adjustment
        elif axis == 'x_max':
            clipping_bounds['x_max'] += adjustment
        elif axis == 'y_min':
            clipping_bounds['y_min'] += adjustment
        elif axis == 'y_max':
            clipping_bounds['y_max'] += adjustment
        elif axis == 'z_min':
            clipping_bounds['z_min'] += adjustment
        elif axis == 'z_max':
            clipping_bounds['z_max'] += adjustment

        # 更新可视化
        update_vis(vis)

    # 颜色更新函数，将指定类别点颜色设置为黄色，其他点设置为黑色
    def update_colors_for_target_class(vis, target_class_id, use_rgb=False):
        """
        根据指定类别ID将该类别的点着色为黄色或使用RGB，其他类别点设置为黑色。

        参数:
        - vis: 可视化对象
        - target_class_id (int): 指定要着色的目标类别编号
        - use_rgb (bool): 是否使用点云的RGB信息，默认为True
        """
        points = np.asarray(pcd.points)
        labels = point_cloud_data[:, 6].astype(np.int64)

        # 创建一个颜色列表，默认设置为黑色
        updated_colors = np.full_like(np.asarray(pcd.colors), [0.2, 0.2, 0.2])

        # 找到目标类别的点，并设置其颜色
        target_class_mask = (labels == target_class_id)

        if use_rgb and point_cloud_data.shape[1] >= 6:
            # 使用点云中的 RGB 信息为目标类别点着色
            updated_colors[target_class_mask] = np.asarray(pcd.colors)[target_class_mask]
        else:
            # 使用黄色（橘色）
            saturation_factor = 0.8  # 降低饱和度的因子，0.7表示降低到70%
            updated_colors[target_class_mask] = [0.96 * saturation_factor, 0.62 * saturation_factor,
                                                 0.26 * saturation_factor]

        # 更新颜色
        pcd.colors = o3d.utility.Vector3dVector(updated_colors)

        # 更新可视化
        update_vis(vis)

    # 截图功能
    def capture_screenshot(vis):
        """
        捕捉当前视图并保存为图像文件。
        """
        screenshot_path = "point_cloud_screenshot.png"  # 指定截图保存路径
        vis.capture_screen_image(screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")

    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)

    # 注册快捷键
    vis.register_key_callback(ord("Q"), lambda vis: adjust_clipping(vis, 'x_min', 0.1))
    vis.register_key_callback(ord("W"), lambda vis: adjust_clipping(vis, 'x_max', -0.1))
    vis.register_key_callback(ord("A"), lambda vis: adjust_clipping(vis, 'y_min', 0.1))
    vis.register_key_callback(ord("S"), lambda vis: adjust_clipping(vis, 'y_max', -0.1))
    vis.register_key_callback(ord("Z"), lambda vis: adjust_clipping(vis, 'z_min', 0.1))
    vis.register_key_callback(ord("X"), lambda vis: adjust_clipping(vis, 'z_max', -0.1))

    # 注册颜色更新快捷键
    if target_class_id is not None:
        vis.register_key_callback(ord("L"), lambda vis: update_colors_for_target_class(vis, target_class_id))

    # 注册截图快捷键
    vis.register_key_callback(ord("C"), lambda vis: capture_screenshot(vis))

    # 提示信息
    print("Press 'Q' to increase x_min")
    print("Press 'W' to decrease x_max")
    print("Press 'A' to increase y_min")
    print("Press 'S' to decrease y_max")
    print("Press 'Z' to increase z_min")
    print("Press 'X' to decrease z_max")
    print("Press 'C' to capture screenshot")
    if target_class_id is not None:
        print(f"Press 'L' to set color for class {target_class_id} points")

    # 启动可视化
    vis.run()
    vis.destroy_window()


def visualize_npy_point_cloud(npy_file_path, with_label=False, dataset=None):
    """
    加载并可视化 .npy 文件中的点云数据，并添加快捷键功能。
    """
    # 加载点云数据
    point_cloud_data = np.load(npy_file_path)

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

    if dataset == 'S3DIS' and with_label:
        # 定义标签颜色
        s3dis_labels = {
            0: [0.65, 0.65, 0.65],  # ceiling (灰色)
            1: [1.00, 0.00, 0.00],  # floor (红色)
            2: [0.00, 1.00, 0.00],  # wall (绿色)
            # 省略其余标签颜色以节省空间...
        }
        labels = point_cloud_data[:, 6].astype(int)
        colors = np.array([s3dis_labels.get(label, [0.5, 0.5, 0.5]) for label in labels])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif point_cloud_data.shape[1] >= 6:
        colors = point_cloud_data[:, 3:6] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # 添加点云到可视化窗口
    vis.add_geometry(pcd)

    # 自定义快捷键功能
    def toggle_wireframe(vis):
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = not opt.mesh_show_wireframe
        return False

    def reset_view(vis):
        vis.reset_view_point(True)
        return False

    def increase_point_size(vis):
        opt = vis.get_render_option()
        opt.point_size += 1
        print(f"Point size: {opt.point_size}")
        return False

    def decrease_point_size(vis):
        opt = vis.get_render_option()
        opt.point_size = max(opt.point_size - 1, 1)
        print(f"Point size: {opt.point_size}")
        return False

    # 注册快捷键
    vis.register_key_callback(ord("W"), toggle_wireframe)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("]"), increase_point_size)
    vis.register_key_callback(ord("["), decrease_point_size)

    # 启动可视化
    vis.run()
    vis.destroy_window()


def show_npy_point_cloud(npy_file_path):
    """
    加载并展示 .npy 文件中的点云数据

    参数:
    npy_file_path (str): 点云数据的 .npy 文件路径
    """
    # 加载 .npy 文件
    point_cloud_data = np.load(npy_file_path)

    # 打印文件内容及其形状
    print("Point Cloud Data:")
    print(point_cloud_data)

    # 显示数据的形状
    print("\nShape of Point Cloud Data:")
    print(point_cloud_data.shape)

    # 打印前五个点以便查看
    print("\nSample Point Cloud Data (first 5 points):")
    print(point_cloud_data[:5])  # 假设每个点是一行


def visualize_h5_point_cloud(h5_file_path):
    # 打开 HDF5 文件
    with h5py.File(h5_file_path, 'r') as f:
        # 读取支持集和查询集的点云数据
        support_ptclouds = f['support_ptclouds'][:]  # 形状 (N, 1, 2048, 9) 这里 N 是支持集的数量
        query_ptclouds = f['query_ptclouds'][:]  # 形状 (M, 2048, 9) 这里 M 是查询集的数量

        # 提取支持集和查询集的坐标 (假设前 3 列是坐标)
        support_points = support_ptclouds[:, 0, :, :3].reshape(-1, 3)  # 形状 (N*2048, 3)
        query_points = query_ptclouds.reshape(-1, 3)  # 形状 (M*2048, 3)

    # 创建 Open3D 点云对象
    support_pcd = o3d.geometry.PointCloud()
    query_pcd = o3d.geometry.PointCloud()

    # 设置点云的坐标
    support_pcd.points = o3d.utility.Vector3dVector(support_points)
    query_pcd.points = o3d.utility.Vector3dVector(query_points)

    # 可视化支持集和查询集点云
    o3d.visualization.draw_plotly([support_pcd, query_pcd],
                                      window_name="Point Clouds Visualization",
                                      width=800,
                                      height=600)


def show_h5_point_cloud(h5_file_path):
    """
    加载并展示 .h5 文件中的点云数据。
    针对测试集h5文件，每个场景存储在一个h5文件中
    文件结构是[query_ptclouds, query_labels, support_ptclouds, support_masks, support_filenames, query_filenames]

    参数:
    h5_file_path (str): .h5 文件路径
    """
    # 打开 .h5 文件
    with h5py.File(h5_file_path, 'r') as f:
        # 打印文件中所有数据集的名称
        print("Datasets in the HDF5 file:")
        for name in f:
            print(f" - {name}: {f[name].shape}")

        # 读取点云、标签和文件名
        query_ptclouds = f['query_ptclouds'][:]  # 形状 (2, 2048, 9)
        query_labels = f['query_labels'][:]  # 形状 (2, 2048)
        support_ptclouds = f['support_ptclouds'][:]  # 形状 (2, 1, 2048, 9)
        support_masks = f['support_masks'][:]  # 形状 (2, 1, 2048)
        sampled_classes = f['sampled_classes'][:]  # 形状 (2,)
        support_filenames = f['support_filenames'][:]  # 形状 (2,)
        query_filenames = f['query_filenames'][:]  # 形状 (2,)

        # 显示数据的形状
        print("\nShapes:")
        print(f"Query Point Clouds shape: {query_ptclouds.shape}")
        print(f"Query Labels shape: {query_labels.shape}")
        print(f"Support Point Clouds shape: {support_ptclouds.shape}")
        print(f"Support Masks shape: {support_masks.shape}")

        # 打印支持集和查询集的文件名
        print("\nSupport Filenames:")
        for idx, filename in enumerate(support_filenames):
            print(f" - Support Filename {idx}: {filename.decode('utf-8')}")  # 解码为字符串

        print("\nQuery Filenames:")
        for idx, filename in enumerate(query_filenames):
            print(f" - Query Filename {idx}: {filename.decode('utf-8')}")  # 解码为字符串

        # 输出对应的支持点云和查询点云的文件名
        print("\nCorrespondence between Query Point Clouds and Filenames:")
        for idx in range(len(query_filenames)):
            print(f"Query Point Cloud {idx} corresponds to {query_filenames[idx].decode('utf-8')}")

        print("\nCorrespondence between Support Point Clouds and Filenames:")
        for idx in range(len(support_filenames)):
            print(f"Support Point Cloud {idx} corresponds to {support_filenames[idx].decode('utf-8')}")

        # **统计 query_labels 中每个类别的点数量**
        print("\nClass-wise Point Counts in Query Point Clouds:")
        for idx, labels in enumerate(query_labels):  # 遍历每个查询点云的标签
            print(f"\nQuery Point Cloud {idx}:")
            unique_classes, counts = torch.unique(torch.tensor(labels), return_counts=True)
            for cls, count in zip(unique_classes, counts):
                print(f" - Class {cls.item()}: {count.item()} points")

        # 打印部分数据以便查看
        print("\nSample data:")
        print("Query Point Clouds Sample (first 5 points from the first point cloud):")
        print(query_ptclouds[0][:5])  # 打印第一个查询点云的前 5 个点

        print("\nQuery Labels Sample (first 5 labels from the first point cloud):")
        print(query_labels[0][:5])  # 打印第一个查询点云的前 5 个标签


def visualize_h5_testfile(file_path, with_label=True, dataset=None):
    """
    读取并使用 Open3D 可视化 HDF5 文件中的查询点云和支持点云，显示每个点云的文件名。

    如果 with_label 为 True，按照标签进行着色；否则使用点云的 RGB 信息着色。

    参数:
    - file_path: str, .h5 文件的路径
    - with_label: bool, 是否根据类别标签对点云着色
    - dataset: str, 数据集名称（如 'S3DIS'）
    """
    with h5py.File(file_path, 'r') as f:
        # 读取点云数据
        query_ptclouds = f['query_ptclouds'][:]  # (num_queries, num_points, 9)
        support_ptclouds = f['support_ptclouds'][:]  # (num_supports, 1, num_points, 9)
        support_masks = f['support_masks'][:]  # (num_supports, num_points)
        query_labels = f['query_labels'][:]  # (num_queries, num_points)
        sampled_classes = f['sampled_classes'][:]  # (num_supports,)
        support_filenames = f['support_filenames'][:]  # (num_supports,)
        query_filenames = f['query_filenames'][:]  # (num_queries,)

        def create_open3d_point_cloud(points, colors):
            """Helper function to create Open3D point cloud."""
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            return point_cloud
            # 定义颜色映射和标签
        if dataset == 'S3DIS':
            s3dis_labels = {
                    0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam',
                    4: 'column', 5: 'window', 6: 'door', 7: 'table',
                    8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'
                }
            num_classes = len(s3dis_labels)
        # 定义颜色映射和标签
        if with_label:
            # 生成颜色映射
            cmap = plt.get_cmap('tab20', num_classes)
            # 可视化查询点云
            print("\nVisualizing Query Point Clouds with Labels...")
            for idx, ptcloud in enumerate(query_ptclouds):
                xyz = ptcloud[:, :3]
                labels = query_labels[idx].astype(int)

                # 查找该 label 属于哪个 sampled_class
                class_colors = []
                for label in labels:
                    if label == 0:
                        # 背景点，设置为黑色
                        class_colors.append([0, 0, 0])
                    else:
                        # 使用 label 映射 sampled_classes 获取真实类别
                        real_class = sampled_classes[label - 1]  # -1 因为 query_labels 范围是 1~n_way
                        color = cmap(real_class / num_classes)
                        class_colors.append([color[0], color[1], color[2]])

                # 将颜色转换为 np 数组
                class_colors = np.array(class_colors)

                # 创建 Open3D 点云对象
                query_pcd = create_open3d_point_cloud(xyz, class_colors)
                o3d.visualization.draw_geometries([query_pcd], window_name=f'Query Point Cloud: {query_filenames[idx].decode()}')
            '''
            # 可视化支持点云
            print("\nVisualizing Support Point Clouds with Masks...")
            for idx, ptcloud in enumerate(support_ptclouds):
                xyz = ptcloud[0, :, :3]
                mask = support_masks[idx].astype(bool)
                class_id = sampled_classes[idx]

                # 使用 mask 设置颜色
                colors = np.where(
                    mask[:, None],  # 为 True 时应用类别颜色，调整为 (num_points, 1)
                    np.array([cmap(class_id / num_classes)[:3]])[None, :],  # 显式调整为 (1, 3)
                    np.array([[0, 0, 0]])[None, :]  # 显式调整为 (1, 3)
                )

                # 创建 Open3D 点云对象
                support_pcd = create_open3d_point_cloud(xyz, colors)
                o3d.visualization.draw_geometries([support_pcd], window_name=f'Support Point Cloud: {support_filenames[idx].decode()} (Class: {class_id})')
            '''
        else:
            # 如果不根据标签着色，使用点云的 RGB 信息
            print("\n with_label=False, Visualizing Point Clouds with RGB Colors...")
            # 可视化查询点云
            for idx, ptcloud in enumerate(query_ptclouds):
                xyz = ptcloud[:, :3]
                rgb = ptcloud[:, 3:6] / 255.0  # 正则化 RGB 值

                # 创建 Open3D 点云对象
                query_pcd = create_open3d_point_cloud(xyz, rgb)
                o3d.visualization.draw_geometries([query_pcd], window_name=f'Query Point Cloud: {query_filenames[idx].decode()}')

            # 可视化支持点云
            for idx, ptcloud in enumerate(support_ptclouds):
                xyz = ptcloud[0, :, :3]
                rgb = ptcloud[0, :, 3:6] / 255.0  # 正则化 RGB 值

                # 创建 Open3D 点云对象
                support_pcd = create_open3d_point_cloud(xyz, rgb)
                o3d.visualization.draw_geometries([support_pcd], window_name=f'Support Point Cloud: {support_filenames[idx].decode()}')


def visualize_h5_predresult(file_path, dataset='s3dis'):
    """
    读取并使用 Open3D 可视化 HDF5 文件中的查询点云、查询标签和预测结果，显示每个点云的文件名。
    并通过颜色条直观展示颜色与类别的对应关系。

    参数:
    - file_path: str, .h5 文件的路径
    - dataset: str, 数据集名称（如 'S3DIS'）
    """
    # 预定义的类别名称字典（0到12）
    if dataset == 's3dis':
        class_to_name = {
            0: "ceiling",
            1: "floor",
            2: "wall",
            3: "beam",
            4: "column",
            5: "window",
            6: "door",
            7: "table",
            8: "chair",
            9: "sofa",
            10: "bookcase",
            11: "board",
            12: "clutter"
        }

    with h5py.File(file_path, 'r') as f:
        # 读取点云数据、标签和预测结果
        query_ptclouds = f['query_ptclouds'][:]  # (num_queries, 9, 2048)
        query_pred = f['query_pred'][:]  # (num_queries, 2048)
        query_labels = f['query_labels'][:]  # (num_queries, 2048)
        query_filenames = f['query_filenames'][:]  # (num_queries,)
        sampled_classes = f['sampled_classes'][:]  # list e.g. [11, 8]

        query_labels_v = torch.tensor(query_labels)
        unique_classes = torch.unique(query_labels_v)

        print(f'block name : {query_filenames}, sampled_classes = {sampled_classes}')
        # 获取并打印每个类别的名称
        for unique_class in unique_classes:
            if unique_class == 0:  # 背景类
                print(f"类 {unique_class}: 背景类")
            else:
                # 从预定义的 class_to_name 字典中获取实际类别名称
                print(f"类 {unique_class}: {class_to_name[sampled_classes[unique_class-1]]}")

        def create_open3d_point_cloud(points, colors):
            """Helper function to create Open3D point cloud."""
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            return point_cloud

        # 定义颜色映射
        if dataset == 's3dis':
            s3dis_labels = {
                0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam',
                4: 'column', 5: 'window', 6: 'door', 7: 'table',
                8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'
            }
            num_classes = len(s3dis_labels)
            background_label = 0  # 背景类
        else:
            s3dis_labels = None
            num_classes = np.max(query_pred) + 1  # 动态获取类别数
            background_label = 0

        cmap = plt.get_cmap('tab20', num_classes)

        def label_to_color(labels):
            """根据标签值映射颜色，背景设为黑色"""
            colors = []
            for label in labels:
                if label == background_label:  # 背景类
                    colors.append([0, 0, 0])  # 黑色
                else:
                    color = cmap(label / num_classes)
                    colors.append([color[0], color[1], color[2]])
            return np.array(colors)

        def display_color_legend():
            """使用 Matplotlib 生成颜色条并显示"""
            fig, ax = plt.subplots(figsize=(8, 2))
            color_list = [cmap(i / num_classes)[:3] for i in range(num_classes)]
            labels = [s3dis_labels[i] if s3dis_labels else f"Class {i}" for i in range(num_classes)]
            ax.imshow([color_list], extent=[0, num_classes, 0, 1], aspect="auto")
            ax.set_xticks(np.arange(num_classes) + 0.5)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
            ax.set_yticks([])
            ax.set_title("Color-to-Class Mapping", fontsize=14)
            plt.tight_layout()
            plt.show()

        # 显示颜色条
        display_color_legend()

        # 打印 query_pred 中存在的类别数字
        unique_classes = np.unique(query_pred)
        print(f"Classes present in query_pred: {unique_classes}")

        # 可视化预测结果
        print("\nVisualizing Query Prediction Results...")
        for idx, ptcloud in enumerate(query_ptclouds):
            xyz = ptcloud[:3, :].T  # 转置以符合 Open3D 的格式
            preds = query_pred[idx].astype(int)  # 预测结果
            pred_colors = label_to_color(preds)

            query_pcd = create_open3d_point_cloud(xyz, pred_colors)
            o3d.visualization.draw_geometries([query_pcd],
                                              window_name=f'Query Prediction: {query_filenames[idx].decode()}')

        # 可视化标签
        print("\nVisualizing Query Ground Truth Labels...")
        for idx, ptcloud in enumerate(query_ptclouds):
            xyz = ptcloud[:3, :].T
            labels = query_labels[idx].astype(int)  # 标签
            label_colors = label_to_color(labels)

            query_pcd = create_open3d_point_cloud(xyz, label_colors)
            o3d.visualization.draw_geometries([query_pcd],
                                              window_name=f'Query Labels: {query_filenames[idx].decode()}')

        # 可视化原始点云（使用 RGB 着色）
        print("\nVisualizing Query Point Clouds with RGB Colors...")
        for idx, ptcloud in enumerate(query_ptclouds):
            xyz = ptcloud[:3, :].T
            rgb = ptcloud[3:6, :].T  # RGB 信息

            query_pcd = create_open3d_point_cloud(xyz, rgb)
            o3d.visualization.draw_geometries([query_pcd],
                                              window_name=f'Query Point Cloud: {query_filenames[idx].decode()}')





def show_h5_pred(h5_file_path, output_csv_path='./pred.csv'):
    """
    加载并展示 .h5 文件中的点云数据，并保存 query_labels 和 query_pred 到 CSV 文件中。

    参数:
    h5_file_path (str): .h5 文件路径
    output_csv_path (str): 输出 CSV 文件路径
    """
    # 打开 .h5 文件
    with h5py.File(h5_file_path, 'r') as f:
        # 打印文件中所有数据集的名称
        print("Datasets in the HDF5 file:")
        for name in f:
            print(f" - {name}: {f[name].shape}")

        # 读取点云、标签和预测值
        query_ptclouds = f['query_ptclouds'][:]  # 形状 (2, 9, 2048)
        query_pred = f['query_pred'][:]  # 形状 (2, 2048)
        query_labels = f['query_labels'][:]  # 形状 (2, 2048)
        query_filenames = f['query_filenames'][:]  # 形状 (2,)
        sampled_classes = f['sampled_classes'][:]
        print(f'Sampled classes = {sampled_classes}')

        # 计算准确率
        query_pred_tensor = torch.tensor(query_pred)
        query_labels_tensor = torch.tensor(query_labels)
        correct = torch.eq(query_pred_tensor, query_labels_tensor).sum().item()  # including background class
        accuracy = correct / (query_labels.shape[0] * query_labels.shape[1])
        print(f'Accuracy = {accuracy:.4f}')

        # 显示数据的形状
        print("\nShapes:")
        print(f"Query Point Clouds shape: {query_ptclouds.shape}")
        print(f"Query Labels shape: {query_labels.shape}")
        print(f"Query Predictions shape: {query_pred.shape}")

        # 输出 unique 标签和预测值
        unique_labels = set(query_labels.flatten())
        unique_preds = set(query_pred.flatten())
        print("\nUnique labels in query_labels:")
        print(sorted(unique_labels))
        print("Unique predictions in query_pred:")
        print(sorted(unique_preds))

        # 保存 query_labels 和 query_pred 到 CSV 文件
        print("\nSaving query_labels and query_pred to CSV file...")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)  # 创建目录（如果不存在）

        with open(output_csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Point_Index', 'Query_Index', 'Label', 'Prediction'])  # 写入表头

            for query_idx in range(query_labels.shape[0]):  # 遍历每个 query
                for point_idx in range(query_labels.shape[1]):  # 遍历每个点
                    label = query_labels[query_idx, point_idx]
                    pred = query_pred[query_idx, point_idx]
                    csv_writer.writerow([point_idx, query_idx, label, pred])

        print(f"Saved query_labels and query_pred to: {output_csv_path}")

        # 打印样例数据
        print("\nSample data:")
        print("Query Labels Sample (first 5 labels from the first point cloud):")
        print(query_labels[0][:5])  # 第一个 query 点云的前 5 个标签
        print("Query Predictions Sample (first 5 predictions from the first point cloud):")
        print(query_pred[0][:5])  # 第一个 query 点云的前 5 个预测值

        # 统计 query_labels 中每个 unique 值的数量
        unique_labels, label_counts = np.unique(query_labels, return_counts=True)
        print("\nGround Truth (GT) Label Distribution:")
        for label, count in zip(unique_labels, label_counts):
            print(f"Label {label}: {count} points")


        # 输出查询点云文件名
        print("\nQuery Filenames:")
        for idx, filename in enumerate(query_filenames):
            print(f" - Query Filename {idx}: {filename.decode('utf-8')}")


def visualize_h5_testset_full_scene(h5_file_path, dataset_root, with_label=True, dataset='S3DIS'):
    """
    从 h5 文件中读取查询点云和标签，并将其与未分块的原始点云进行可视化。
    query_labels 的点使用标签上色，其他点用原始的 RGB 信息表示。

    参数:
    - h5_file_path: str, h5 文件的路径
    - dataset_root: str, 原始点云文件 (.npy) 所在的根目录路径
    - with_label: bool, 是否对标签点上色
    - dataset: str, 数据集名称（默认为 'S3DIS'）
    """
    with h5py.File(h5_file_path, 'r') as f:
        query_ptclouds = f['query_ptclouds'][:]  # (num_queries, num_points, 9)
        query_labels = f['query_labels'][:]  # (num_queries, num_points)
        query_filenames = f['query_filenames'][:]  # (num_queries,)
        sampled_classes = f['sampled_classes'][:]  # (num_supports,)

    colors_map = {}
    labels_map = {}

    if dataset == 'S3DIS':
        colors_map = {
            1: [255, 0, 0],    # beam
            2: [0, 255, 0],    # wall
            3: [0, 0, 255],    # column
            4: [255, 255, 0],  # floor
            5: [255, 0, 255],  # ceiling
            6: [0, 255, 255],  # window
            7: [255, 128, 0],  # door
            8: [0, 128, 255],  # table
            9: [128, 0, 255],  # chair
            10: [128, 255, 0], # sofa
            11: [0, 128, 0],   # bookcase
            12: [128, 128, 0], # clutter
        }

        labels_map = {
            0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam',
            4: 'column', 5: 'window', 6: 'door', 7: 'table',
            8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'
        }

    for idx, query_filename_bytes in enumerate(query_filenames):
        query_filename = query_filename_bytes.decode('utf-8')

        # 提取场景名称和块号
        scene_name = query_filename.split("_[")[1].split('_block')[0].strip("'")
        block_number = int(query_filename.split('_block_')[-1].split("'")[0])
        h5_query_ptcloud = query_ptclouds[idx, :, :3]  # 当前块点云
        h5_query_rgb = query_ptclouds[idx, :, 3:6] / 255.0  # 当前块 RGB 颜色
        h5_query_labels = query_labels[idx]  # 当前块标签

        print(f"Processing query {idx + 1}/{len(query_filenames)}: {query_filename}")
        print(f"Scene name: {scene_name}, Block number: {block_number}")

        # 当前块上色
        current_block_colors = np.zeros_like(h5_query_rgb)  # 初始化为全零的颜色数组
        if with_label:
            for i, pt in enumerate(h5_query_ptcloud):
                label = h5_query_labels[i]
                if label != 0:  # 只处理非背景点
                    real_class = sampled_classes[label - 1]
                    color = colors_map.get(real_class, [0, 0, 0])  # 获取颜色，默认黑色
                    current_block_colors[i] = np.array(color) / 255.0
                else:  # 对于背景点，使用原始 RGB 颜色
                    current_block_colors[i] = h5_query_rgb[i]

        # 加载 dataset_root 下同场景的其他块
        block_files = glob.glob(os.path.join(dataset_root, f'{scene_name}_block_*.npy'))
        other_blocks = []

        for block_file in block_files:
            other_block_number = int(os.path.basename(block_file).split('_block_')[-1].split('.')[0])
            if other_block_number != block_number:  # 排除当前块
                print(f"Loading block: {block_file}")
                other_block_data = np.load(block_file)

                # 提取 xyz 和 rgb 数据
                other_block_xyz = other_block_data[:, :3]  # 提取其他块的坐标
                other_block_rgb = other_block_data[:, 3:6] / 255.0  # 提取其他块的颜色并归一化

                # 拼接 xyz 和颜色信息
                other_block_combined = np.hstack((other_block_xyz, other_block_rgb))
                other_blocks.append(other_block_combined)

        # 拼接当前块和其他块
        blocks_to_combine = [np.hstack((h5_query_ptcloud, current_block_colors))] + other_blocks
        full_scene_data = np.concatenate(blocks_to_combine, axis=0)

        # 分离拼接后的点云和颜色
        full_scene_xyz = full_scene_data[:, :3]
        full_scene_colors = full_scene_data[:, 3:6]

        # 调试信息
        print(f"Total points in scene after combining: {full_scene_xyz.shape[0]}")
        print(f"Full scene shape: {full_scene_data.shape}")

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(full_scene_xyz)
        pcd.colors = o3d.utility.Vector3dVector(full_scene_colors)

        # 可视化
        o3d.visualization.draw_geometries([pcd],
                                          window_name=f'Full Scene: {scene_name} | Block {block_number}')



def visualize_h5_result_full_scene(result_root, testset_root, dataset_root, with_label=True, dataset='S3DIS'):
    """
    从 result_root 读取结果文件并与 testset_root 中的原始点云进行可视化。
    从 result 文件中提取 query_pred 作为标签，并从 h5 文件中提取点云数据。

    参数:
    - result_root: str, 结果文件夹路径
    - testset_root: str, 原始点云文件所在的根目录路径
    - dataset_root: str, 数据集名称（默认为 'S3DIS'）
    - with_label: bool, 是否对标签点上色
    """

    # 从 result_root 获取所有的 segmentation_results_*.h5 文件
    result_files = glob.glob(os.path.join(result_root, 'segmentation_results_*.h5'))

    for result_file in result_files:
        # 提取基础文件名（取 '*' 部分）
        base_filename = os.path.basename(result_file).replace('segmentation_results_', '').replace('.h5', '')

        # 找到对应的 testset_root 中的 h5 文件
        h5_file_path = os.path.join(testset_root, f'{base_filename}.h5')
        if not os.path.exists(h5_file_path):
            print(f"Warning: Corresponding h5 file not found for {base_filename}. Skipping this result.")
            continue

        # 读取 result 文件中的 query_pred
        with h5py.File(result_file, 'r') as f:
            query_pred = f['query_pred'][:]  # (num_queries, num_points)

        # 读取对应的 h5 文件中的 query_ptclouds、query_filenames、sampled_classes
        with h5py.File(h5_file_path, 'r') as f:
            query_ptclouds = f['query_ptclouds'][:]  # (num_queries, num_points, 9)
            query_filenames = f['query_filenames'][:]  # (num_queries,)
            sampled_classes = f['sampled_classes'][:]  # (num_supports,)

        colors_map = {}
        labels_map = {}

        if dataset == 'S3DIS':
            colors_map = {
                1: [255, 0, 0],    # beam
                2: [0, 255, 0],    # wall
                3: [0, 0, 255],    # column
                4: [255, 255, 0],  # floor
                5: [255, 0, 255],  # ceiling
                6: [0, 255, 255],  # window
                7: [255, 128, 0],  # door
                8: [0, 128, 255],  # table
                9: [128, 0, 255],  # chair
                10: [128, 255, 0], # sofa
                11: [0, 128, 0],   # bookcase
                12: [128, 128, 0], # clutter
            }

            labels_map = {
                0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam',
                4: 'column', 5: 'window', 6: 'door', 7: 'table',
                8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'
            }

        for idx, query_filename_bytes in enumerate(query_filenames):
            query_filename = query_filename_bytes.decode('utf-8')

            # 提取场景名称和块号
            scene_name = query_filename.split("_[")[1].split('_block')[0].strip("'")
            block_number = int(query_filename.split('_block_')[-1].split("'")[0])
            h5_query_ptcloud = query_ptclouds[idx, :, :3]  # 当前块点云
            h5_query_rgb = query_ptclouds[idx, :, 3:6] / 255.0  # 当前块 RGB 颜色
            h5_query_labels = query_pred[idx]  # 当前块标签（从 result 文件中读取）

            print(f"Processing query {idx + 1}/{len(query_filenames)}: {query_filename}")
            print(f"Scene name: {scene_name}, Block number: {block_number}")

            # 当前块上色
            current_block_colors = np.zeros_like(h5_query_rgb)  # 初始化为全零的颜色数组
            if with_label:
                for i, pt in enumerate(h5_query_ptcloud):
                    label = h5_query_labels[i]
                    if label != 0:  # 只处理非背景点
                        real_class = sampled_classes[label - 1]
                        color = colors_map.get(real_class, [0, 0, 0])  # 获取颜色，默认黑色
                        current_block_colors[i] = np.array(color) / 255.0
                    else:  # 对于背景点，使用原始 RGB 颜色
                        current_block_colors[i] = h5_query_rgb[i]

            # 加载 dataset_root 下同场景的其他块
            block_files = glob.glob(os.path.join(dataset_root, f'{scene_name}_block_*.npy'))
            other_blocks = []

            for block_file in block_files:
                other_block_number = int(os.path.basename(block_file).split('_block_')[-1].split('.')[0])
                if other_block_number != block_number:  # 排除当前块
                    print(f"Loading block: {block_file}")
                    other_block_data = np.load(block_file)

                    # 提取 xyz 和 rgb 数据
                    other_block_xyz = other_block_data[:, :3]  # 提取其他块的坐标
                    other_block_rgb = other_block_data[:, 3:6] / 255.0  # 提取其他块的颜色并归一化

                    # 拼接 xyz 和颜色信息
                    other_block_combined = np.hstack((other_block_xyz, other_block_rgb))
                    other_blocks.append(other_block_combined)

            # 拼接当前块和其他块
            blocks_to_combine = [np.hstack((h5_query_ptcloud, current_block_colors))] + other_blocks
            full_scene_data = np.concatenate(blocks_to_combine, axis=0)

            # 分离拼接后的点云和颜色
            full_scene_xyz = full_scene_data[:, :3]
            full_scene_colors = full_scene_data[:, 3:6]

            # 调试信息
            print(f"Total points in scene after combining: {full_scene_xyz.shape[0]}")
            print(f"Full scene shape: {full_scene_data.shape}")

            # 创建 Open3D 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(full_scene_xyz)
            pcd.colors = o3d.utility.Vector3dVector(full_scene_colors)

            # 可视化
            o3d.visualization.draw_geometries([pcd],
                                              window_name=f'Full Scene: {scene_name} | Block {block_number}')


def visual_result(result_file_path, test_file_path, result_with_label=True, test_with_label=True, dataset=None):
    print("Visualizing result file...")
    visualize_h5_predresult(result_file_path, with_label=result_with_label, dataset=dataset)


def get_scans_for_class(class_id, pkl_file_path):
    """
    从指定的 pkl 文件中获取对应类别号的场景信息。

    参数:
    - class_id (int): 类别号。
    - pkl_file_path (str): 存储 class2scans 的 .pkl 文件路径。

    返回:
    - block_names (list): 对应类别号的块名称列表。
    """
    if not os.path.exists(pkl_file_path):
        print(f"Error: The pkl file '{pkl_file_path}' does not exist.")
        return []

    try:
        # 加载 pkl 文件
        with open(pkl_file_path, 'rb') as f:
            class2scans = pickle.load(f)

        # 检查类别号是否在字典中
        if class_id in class2scans:
            block_names = class2scans[class_id]
            print(f"Blocks for class {class_id}: {block_names}")
            return block_names
        else:
            print(f"Class ID {class_id} not found in the dictionary.")
            return []
    except Exception as e:
        print(f"An error occurred while reading the pkl file: {e}")
        return []


def extract_scene_names(block_names):
    """
    从块名称中提取场景名称。

    参数:
    - block_names (list): 包含块名称的列表（如 'scene0554_00_block_4_row1_col1'）。

    返回:
    - scene_names (list): 提取出的场景名称（如 'scene0554_00.npy'）。
    """
    scene_names = list({block.split('_block')[0] + '.npy' for block in block_names})
    return scene_names


def visualize_scenes(scene_dir, scene_names):
    """
    可视化场景点云。

    参数:
    - scene_dir (str): 场景 .npy 文件所在的目录。
    - scene_names (list): 场景文件名列表（如 ['scene0554_00.npy', 'scene0613_02.npy']）。
    """
    for scene_name in scene_names:
        scene_path = os.path.join(scene_dir, scene_name)
        if os.path.exists(scene_path):
            print(f"Visualizing: {scene_path}")
            visualize_npy_point_cloud(scene_path)
        else:
            print(f"Scene file not found: {scene_path}")

