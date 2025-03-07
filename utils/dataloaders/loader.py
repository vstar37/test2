import os
import random
import math
import glob
import numpy as np
import h5py as h5
import transforms3d
from itertools import  combinations

import torch
from torch.utils.data import Dataset
from config import Config
import ast
config = Config()

def sample_K_pointclouds(data_path, num_point, pc_attribs, pc_augm, pc_augm_config,
                         scan_names, sampled_class, sampled_classes, is_support=False):
    '''Sample K pointclouds and the corresponding labels for one class (one_way)'''
    ptclouds = []
    labels = []
    xyz_mins = []  # 用于存储每个样本的 xyz_min

    for scan_name in scan_names:
        ptcloud, label, xyz_min = sample_pointcloud(data_path, num_point, pc_attribs, pc_augm,pc_augm_config ,
                                                    scan_name, sampled_classes, sampled_class, support=is_support)
        ptclouds.append(ptcloud)
        labels.append(label)
        xyz_mins.append(xyz_min)  # 追加每次采样得到的 xyz_min

    # 对每次采样得到的点云、标签和 xyz_min 进行 stack 操作
    ptclouds = np.stack(ptclouds, axis=0)
    labels = np.stack(labels, axis=0)
    xyz_mins = np.stack(xyz_mins, axis=0)  # 将所有 xyz_min 进行 stack

    return ptclouds, labels, xyz_mins  # 返回采样后的点云、标签以及 xyz_min


def sample_neighbor_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config):
    # encoder load neighbor pointcloud

    data = np.load(data_path)
    N = data.shape[0]  # number of points in this scan

    sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point))
    np.random.shuffle(sampled_point_inds)
    data = data[sampled_point_inds]  # 获取随机采样点的数据

    xyz = data[:, 0:3]
    rgb = data[:, 3:6]

    xyz_min = np.amin(xyz, axis=0)  # 计算 xyz_min
    xyz -= xyz_min  # 归一化点云坐标

    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)

    if 'XYZ' in pc_attribs:
        xyz_min_aug = np.amin(xyz, axis=0)  # Augmentation 后的最小值
        XYZ = xyz - xyz_min_aug  # 再次归一化
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ / xyz_max

    ptcloud = []
    if 'xyz' in pc_attribs: ptcloud.append(xyz)
    if 'rgb' in pc_attribs: ptcloud.append(rgb / 255.)
    if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1)
    ptcloud = np.transpose(ptcloud)  # 变为 (9, n)
    ptcloud = torch.tensor(ptcloud).float().to('cuda')

    return ptcloud, xyz_min  # 返回 xyz_min 以便后续复原


def sample_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name,
                      sampled_classes, sampled_class=0, support=False, random_sample=False):
    sampled_classes = list(sampled_classes)

    # 匹配新的文件名格式，使用 glob 方式查找对应的文件
    scan_file_pattern = os.path.join(data_path, 'data', '%s.npy' % scan_name)
    matching_files = glob.glob(scan_file_pattern)

    if not matching_files:
        raise FileNotFoundError(f"No matching files found for scan_name: {scan_name} with pattern {scan_file_pattern}")

    # 假设 scan_name 的文件匹配到唯一一个文件
    data_file = matching_files[0]
    data = np.load(data_file)
    N = data.shape[0]  # number of points in this scan

    if random_sample:
        sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point))
    else:
        # If this point cloud is for support/query set, make sure that the sampled points contain target class
        valid_point_inds = np.nonzero(data[:, 6] == sampled_class)[
            0]  # indices of points belonging to the sampled class

        if N < num_point:
            sampled_valid_point_num = len(valid_point_inds)
        else:
            valid_ratio = len(valid_point_inds) / float(N)
            sampled_valid_point_num = int(valid_ratio * num_point)

        sampled_valid_point_inds = np.random.choice(valid_point_inds, sampled_valid_point_num, replace=False)
        sampled_other_point_inds = np.random.choice(np.arange(N), num_point - sampled_valid_point_num,
                                                    replace=(N < num_point))
        sampled_point_inds = np.concatenate([sampled_valid_point_inds, sampled_other_point_inds])


    np.random.shuffle(sampled_point_inds)
    data = data[sampled_point_inds]  # 获取随机采样点的数据

    xyz = data[:, 0:3]
    rgb = data[:, 3:6]
    labels = data[:, 6].astype(np.int64)

    xyz_min = np.amin(xyz, axis=0)  # 计算 xyz_min
    xyz -= xyz_min  # 归一化点云坐标

    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)

    if 'XYZ' in pc_attribs:
        xyz_min_aug = np.amin(xyz, axis=0)  # Augmentation 后的最小值
        XYZ = xyz - xyz_min_aug  # 再次归一化
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ / xyz_max

    ptcloud = []
    if 'xyz' in pc_attribs: ptcloud.append(xyz)
    if 'rgb' in pc_attribs: ptcloud.append(rgb / 255.)
    if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1)

    if support:
        groundtruth = labels == sampled_class
    else:
        groundtruth = np.zeros_like(labels)
        for i, label in enumerate(labels):
            if label in sampled_classes:
                groundtruth[i] = sampled_classes.index(label) + 1

    return ptcloud, groundtruth, xyz_min  # 返回 xyz_min 以便后续复原


def augment_pointcloud(P, pc_augm_config):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_config['scale'] > 1:
        s = random.uniform(1 / pc_augm_config['scale'], pc_augm_config['scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_config['rot'] == 1:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # z=upright assumption
    if pc_augm_config['mirror_prob'] > 0:  # mirroring x&y, not z
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), M)
    P[:, :3] = np.dot(P[:, :3], M.T)
    if pc_augm_config['shift'] > 0:
        shift = np.random.uniform(-pc_augm_config['shift'], pc_augm_config['shift'], 3)
        P[:, :3] += shift
    if pc_augm_config['jitter']:
        sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P


def reverse_augment_pointcloud(P, pc_augm_config):
    """ Reverse augmentation for point cloud. This should undo the operations done in augment_pointcloud """

    # 保存原设备
    device = P.device if isinstance(P, torch.Tensor) else torch.device('cpu')

    # 将P从Tensor转换为NumPy数组进行增广逆操作
    P = P.cpu().numpy() if isinstance(P, torch.Tensor) else P  # 如果是Tensor，移动到CPU并转为NumPy

    # 1. 逆平移操作
    if pc_augm_config['shift'] > 0:
        shift = np.random.uniform(-pc_augm_config['shift'], pc_augm_config['shift'], 3)
        P[:, :3] -= shift  # 逆平移

    # 2. 逆镜像操作
    if pc_augm_config['mirror_prob'] > 0:  # 检查是否有镜像
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            P[:, 0] = -P[:, 0]  # 恢复镜像
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            P[:, 1] = -P[:, 1]  # 恢复镜像

    # 3. 逆旋转操作
    if pc_augm_config['rot'] == 1:
        # 逆旋转：旋转矩阵的转置
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], -angle), np.eye(3))
        P[:, :3] = np.dot(P[:, :3], M.T)  # 逆旋转

    # 4. 逆缩放操作
    if pc_augm_config['scale'] > 1:
        s = random.uniform(1 / pc_augm_config['scale'], pc_augm_config['scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(1 / s), np.eye(3))  # 逆缩放
        P[:, :3] = np.dot(P[:, :3], M.T)

    # 将结果转回到原始设备上（GPU或CPU）
    return torch.tensor(P).to(device)  # 确保返回到原设备


class MyDataset(Dataset):
    def __init__(self, data_path, dataset_name, cvfold=0, num_episode=50000, n_way=3, k_shot=5, n_queries=1,
                 phase=None, mode='train', num_point=4096, pc_attribs='xyz', pc_augm=False, pc_augm_config=None):
        super(MyDataset).__init__()
        self.data_path = data_path
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_queries = n_queries
        self.num_episode = num_episode
        self.phase = phase
        self.mode = mode
        self.num_point = num_point
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        if dataset_name == 's3dis':
            from utils.dataloaders.s3dis import S3DISDataset
            self.dataset = S3DISDataset(cvfold, data_path)
        elif dataset_name == 'scannet':
            from utils.dataloaders.scannet import ScanNetDataset
            self.dataset = ScanNetDataset(cvfold, data_path)
        else:
            raise NotImplementedError('Unknown dataset %s!' % dataset_name)

        if mode == 'train':
            self.classes = np.array(self.dataset.train_classes)
        elif mode == 'test':
            self.classes = np.array(self.dataset.test_classes)
        else:
            raise NotImplementedError('Unkown mode %s! [Options: train/test]' % mode)

        print('MODE: {0} | Classes: {1}'.format(mode, self.classes))
        self.class2scans = self.dataset.class2scans
        self.class2scenes = self.dataset.class2scenes


    def __len__(self):
        return self.num_episode
    # 传入 episode_index 和 n way classes 列表
    def __getitem__(self, index, n_way_classes=None):
        # 指定类别
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
            print("sampled_classes = {}".format(sampled_classes))
        else:
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)

        support_ptclouds, support_masks, query_ptclouds, query_labels, support_filenames, query_filenames, support_xyz_mins, query_xyz_mins = self.generate_one_episode(sampled_classes)

        return support_ptclouds.astype(np.float32), \
               support_masks.astype(np.int32), \
               query_ptclouds.astype(np.float32), \
               query_labels.astype(np.int64), \
               support_filenames, query_filenames, \
               support_xyz_mins, query_xyz_mins, \
               sampled_classes.astype(np.int32)

    def generate_one_episode(self, sampled_classes):
        support_ptclouds = []
        support_masks = []
        query_ptclouds = []
        query_labels = []

        support_filenames = []  # 用于存储支持集文件名
        query_filenames = []  # 用于存储查询集文件名

        support_xyz_mins_all = []  # 用于存储每个 support 块的 xyz_min
        query_xyz_mins_all = []  # 用于存储每个 query 块的 xyz_min

        black_list = []  # 防止重复采样的scan names

        # 一次处理一个类别 one way
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            if len(black_list) != 0:
                all_scannames = [x for x in all_scannames if x not in black_list]

            # 随机选择 k_shot + n_queries 个场景
            selected_scannames = np.random.choice(all_scannames, self.k_shot + self.n_queries, replace=False)
            black_list.extend(selected_scannames)

            # 前 n_queries 个作为查询集，其余是支持集
            query_scannames = selected_scannames[:self.n_queries]
            support_scannames = selected_scannames[self.n_queries:]

            # 采样查询集的点云和标签
            query_ptclouds_one_way, query_labels_one_way, query_xyz_min_one_way = sample_K_pointclouds(
                self.data_path,
                self.num_point,
                self.pc_attribs,
                self.pc_augm,
                self.pc_augm_config,
                query_scannames,
                sampled_class,
                sampled_classes,
                is_support=False
            )

            # 采样支持集的点云和mask
            support_ptclouds_one_way, support_masks_one_way, support_xyz_min_one_way = sample_K_pointclouds(
                self.data_path,
                self.num_point,
                self.pc_attribs,
                self.pc_augm,
                self.pc_augm_config,
                support_scannames,
                sampled_class,
                sampled_classes,
                is_support=True
            )

            # 将点云数据和标签加入对应列表
            query_ptclouds.append(query_ptclouds_one_way)
            query_labels.append(query_labels_one_way)
            support_ptclouds.append(support_ptclouds_one_way)
            support_masks.append(support_masks_one_way)

            # 生成文件名（支持集和查询集）
            support_filename = f'support_{sampled_class}_{support_scannames}'
            query_filename = f'query_{sampled_class}_{query_scannames}'

            # 将文件名加入列表
            support_filenames.append(support_filename)
            query_filenames.append(query_filename)

            # 保存每个类别的 xyz_min
            support_xyz_mins_all.append(support_xyz_min_one_way)
            query_xyz_mins_all.append(query_xyz_min_one_way)

        # 将支持集和查询集拼接
        support_ptclouds = np.stack(support_ptclouds, axis=0)
        support_masks = np.stack(support_masks, axis=0)
        query_ptclouds = np.concatenate(query_ptclouds, axis=0)
        query_labels = np.concatenate(query_labels, axis=0)

        # 拼接原始坐标
        support_xyz_mins = np.concatenate(support_xyz_mins_all, axis=0)
        query_xyz_mins = np.concatenate(query_xyz_mins_all, axis=0)

        # 返回所有数据和文件名
        return support_ptclouds, support_masks, query_ptclouds, query_labels, support_filenames, query_filenames, support_xyz_mins, query_xyz_mins


class MyDatasetVIS(MyDataset):
    def __init__(self, data_path, dataset_name, cvfold=0, num_episode=50000, n_way=3, k_shot=5, n_queries=1,
                 phase=None, mode='train', num_point=4096, pc_attribs='xyz', pc_augm=False, pc_augm_config=None):
        super().__init__(data_path, dataset_name, cvfold, num_episode, n_way, k_shot, n_queries,
                         phase, mode, num_point, pc_attribs, pc_augm, pc_augm_config)

    def __getitem__(self, index, n_way_classes=None):
        # 指定类别
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
            print("sampled_classes = {}".format(sampled_classes))
        else:
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)

        # k个支持，同一场景下的全部查询
        support_ptclouds, support_masks, query_ptclouds, query_labels, support_filenames, query_filenames, support_xyz_mins, query_xyz_mins = self.generate_one_episode(sampled_classes)

        return support_ptclouds.astype(np.float32), \
               support_masks.astype(np.int32), \
               query_ptclouds.astype(np.float32), \
               query_labels.astype(np.int64), \
               support_filenames, query_filenames, \
               support_xyz_mins, query_xyz_mins, \
            sampled_classes # debug for visTrain

    def generate_one_episode(self, sampled_classes):
        support_ptclouds = []
        support_masks = []
        query_ptclouds = []
        query_labels = []

        support_filenames = []  # 用于存储支持集文件名
        query_filenames = []  # 用于存储查询集文件名

        support_xyz_mins_all = []  # 用于存储每个 support 块的 xyz_min
        query_xyz_mins_all = []  # 用于存储每个 query 块的 xyz_min

        black_list_scan = []  # 防止重复采样的scan names
        black_list_scene = []  # 防止重复采样的scan names

        # 一次处理一个类别 one way
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            all_scenenames = self.class2scenes[sampled_class].copy()

            if len(black_list_scene) != 0:
                all_scenenames = [x for x in all_scenenames if x not in black_list_scene]  # 不重复的，包含目标类别的场景

            if len(black_list_scan) != 0:
                all_scannames = [x for x in all_scannames if x not in black_list_scan]  # 不重复的，包含目标类别的块

            # 随机选择 n_queries 个场景作为查询集场景
            selected_scenenames = np.random.choice(all_scenenames, self.n_queries, replace=False)
            black_list_scene.extend(selected_scenenames)

            # 获取每个 selected_scene 对应的所有 scans 作为 query_scannames
            query_scannames = [
                scan for scan in all_scannames if any(scene in scan for scene in selected_scenenames)
            ]  # 找到当前查询场景下所有包含目标类别的块

            query_scannames = list(set(query_scannames))  # 去重，防止重复
            # query_scannames = list(set(query_scannames))[:self.n_queries]  # 限制为 n_queries 个块 debug


            black_list_scan.extend(query_scannames)  # 将查询集块加入黑名单

            # 支持集采样逻辑
            # 从 all_scannames 中排除与查询集场景相关的块
            support_candidates = [
                scan for scan in all_scannames
                if not any(scene in scan for scene in selected_scenenames) and scan not in black_list_scan
            ]

            # 如果可用块不足 k_shot，抛出异常
            if len(support_candidates) < self.k_shot:
                raise ValueError("Not enough scans available for support set sampling.")

            # 随机选择 k_shot 个支持集扫描块
            selected_scannames = np.random.choice(support_candidates, self.k_shot, replace=False)
            black_list_scan.extend(selected_scannames)

            # 采样查询集的点云和标签 （一个场景下的所有query块)
            query_ptclouds_one_way, query_labels_one_way, query_xyz_min_one_way = sample_K_pointclouds(
                self.data_path,
                self.num_point,
                self.pc_attribs,
                self.pc_augm,
                self.pc_augm_config,
                query_scannames,
                sampled_class,
                sampled_classes,
                is_support=False
            )

            # 采样支持集的点云和mask
            support_ptclouds_one_way, support_masks_one_way, support_xyz_min_one_way = sample_K_pointclouds(
                self.data_path,
                self.num_point,
                self.pc_attribs,
                self.pc_augm,
                self.pc_augm_config,
                selected_scannames,
                sampled_class,
                sampled_classes,
                is_support=True
            )

            # 将点云数据和标签加入对应列表
            query_ptclouds.append(query_ptclouds_one_way)
            query_labels.append(query_labels_one_way)
            support_ptclouds.append(support_ptclouds_one_way)
            support_masks.append(support_masks_one_way)

            # 生成文件名（支持集和查询集）
            support_filename = f'support_{sampled_class}_{selected_scannames}'
            query_filename = f'query_{sampled_class}_{query_scannames}'

            # 将文件名加入列表
            support_filenames.append(support_filename)
            query_filenames.append(query_filename)

            # 保存每个类别的 xyz_min
            support_xyz_mins_all.append(support_xyz_min_one_way)
            query_xyz_mins_all.append(query_xyz_min_one_way)

        # 将支持集和查询集拼接
        support_ptclouds = np.stack(support_ptclouds, axis=0)
        support_masks = np.stack(support_masks, axis=0)
        query_ptclouds = np.concatenate(query_ptclouds, axis=0)
        query_labels = np.concatenate(query_labels, axis=0)

        # 拼接原始坐标
        support_xyz_mins = np.concatenate(support_xyz_mins_all, axis=0)
        query_xyz_mins = np.concatenate(query_xyz_mins_all, axis=0)


        # 返回所有数据和文件名
        return support_ptclouds, support_masks, query_ptclouds, query_labels, support_filenames, query_filenames, support_xyz_mins, query_xyz_mins

def batch_train_task_collate(batch):
    batch_support_ptclouds, batch_support_masks, batch_query_ptclouds, batch_query_labels, \
    batch_support_filenames, batch_query_filenames, \
    batch_support_xyz_mins, batch_query_xyz_mins, batch_sampled_classes = batch[0]

    # 将点云数据转换为 torch tensor 并做转置处理
    data = [
        torch.from_numpy(batch_support_ptclouds).transpose(2, 3),  # 支持集点云数据
        torch.from_numpy(batch_support_masks),                     # 支持集掩码
        torch.from_numpy(batch_query_ptclouds).transpose(1, 2),    # 查询集点云数据
        torch.from_numpy(batch_query_labels.astype(np.int64)),     # 查询集标签
        batch_support_filenames,                                  # 支持集文件名
        batch_query_filenames,                                    # 查询集文件名
        torch.from_numpy(batch_support_xyz_mins),                 # 支持集 xyz_min
        torch.from_numpy(batch_query_xyz_mins)                    # 查询集 xyz_min
    ]

    return data, batch_sampled_classes


################################################ Static Testing Dataset ################################################

class MyTestDataset(Dataset):
    def __init__(self, data_path, dataset_name, cvfold=0, num_episode_per_comb=100, n_way=3, k_shot=5, n_queries=1,
                       num_point=4096, pc_attribs='xyz', mode='valid'):
        super(MyTestDataset).__init__()

        # 获取 n way k shot 数据集
        dataset = MyDataset(data_path, dataset_name, cvfold=cvfold, n_way=n_way, k_shot=k_shot, n_queries=n_queries,
                            mode='test', num_point=num_point, pc_attribs=pc_attribs, pc_augm=False)

        vis_dataset = MyDatasetVIS(data_path, dataset_name, cvfold=cvfold, n_way=n_way, k_shot=k_shot, n_queries=n_queries,
                            mode='test', num_point=num_point, pc_attribs=pc_attribs, pc_augm=False)


        self.classes = dataset.classes

        if mode == 'valid':
            test_data_path = os.path.join(data_path, 'S_%d_N_%d_K_%d_episodes_%d_pts_%d' % (
                                                    cvfold, n_way, k_shot, num_episode_per_comb, num_point))
        elif mode == 'test':
            if config.vis_test:
                test_data_path = os.path.join(data_path, 'S_%d_N_%d_K_%d_test_vis_episodes_%d_pts_%d' % (
                    cvfold, n_way, k_shot, num_episode_per_comb, num_point))
            else:
                test_data_path = os.path.join(data_path, 'S_%d_N_%d_K_%d_test_episodes_%d_pts_%d' % (
                                                    cvfold, n_way, k_shot, num_episode_per_comb, num_point))
        else:
            raise NotImplementedError('Mode (%s) is unknown!' %mode)

        # 如果 fold 或者 类别换了 需要删除生成的测试数据集 再重新生成。生成过程选用的场景具有一定随机性，每次运行eval脚本都会读取这个路径。
        if os.path.exists(test_data_path):
            self.file_names = glob.glob(os.path.join(test_data_path, '*.h5'))
            self.num_episode = len(self.file_names)
        # 生成测试数据集，将测试类别进行排列组合，
        elif mode == 'test':
            if not config.vis_test:
                # 考虑启用 更便于 可视化的命名
                print('Test dataset (%s) does not exist...\n Constructing...' % test_data_path)
                os.mkdir(test_data_path)
                class_comb = list(combinations(self.classes, n_way))  # [(),(),(),...]
                self.num_episode = len(class_comb) * num_episode_per_comb
                episode_ind = 0
                self.file_names = []
                # 每100个h5文件来自于一个dataset episode
                for sampled_classes in class_comb:
                    sampled_classes = list(sampled_classes)
                    for i in range(num_episode_per_comb):
                        # 获取支持集和查询集的点云、标签和文件名
                        data = dataset.__getitem__(
                            episode_ind, sampled_classes)
                        out_filename = os.path.join(test_data_path, '%d.h5' % episode_ind)
                        # 保存支持集和查询集数据
                        write_episode(out_filename, data)
                        # 将生成的文件名添加到列表中
                        self.file_names.append(out_filename)
                        episode_ind += 1
                self.num_episode = len(self.file_names)
            else:
                # 从config读取 指定目标类别
                print('Test dataset (%s) does not exist...\n Constructing...' % test_data_path)
                os.mkdir(test_data_path)
                class_comb = list(combinations(self.classes, n_way))  # [(), (), (), ...]
                self.num_episode = 0
                episode_ind = 0
                self.file_names = []

                for sampled_classes in class_comb:
                    sampled_classes = list(sampled_classes)
                    data = vis_dataset.__getitem__(episode_ind, sampled_classes)
                    support_ptclouds, support_masks, query_ptclouds, query_labels, support_filenames, query_filenames, support_xyz_mins, query_xyz_mins = data

                    # 处理查询数据
                    for fn in query_filenames:
                        # 切分字符串，获取前缀、块列表和后缀
                        prefix, _, suffix = fn.partition('[')
                        list_str = '[' + suffix  # 保留 '[' 并去除后面的部分
                        block_list = ast.literal_eval(list_str)  # 转换为实际的列表

                        # 逐个处理块
                        for i, block in enumerate(block_list):
                            # 当前查询数据
                            current_query_ptcloud = query_ptclouds[i, :, :]
                            current_query_label = query_labels[i]
                            current_query_filename = f"{prefix}[{repr(block)}]"
                            current_query_xyz_min = query_xyz_mins[i]

                            # 构建输出文件路径
                            out_filename = os.path.join(test_data_path, '{}_vid.h5'.format(block))

                            # 保存支持集和当前查询集的数据
                            with h5.File(out_filename, 'w') as data_file:
                                # 写入支持集和查询集的点云、掩膜、标签和类
                                data_file.create_dataset('support_ptclouds', data=support_ptclouds, dtype='float32')
                                data_file.create_dataset('support_masks', data=support_masks, dtype='int32')
                                data_file.create_dataset('query_ptclouds', data=current_query_ptcloud[np.newaxis, ...],
                                                         dtype='float32')
                                data_file.create_dataset('query_labels', data=current_query_label[np.newaxis, ...],
                                                         dtype='int64')
                                data_file.create_dataset('sampled_classes', data=sampled_classes, dtype='int32')

                                # 保存支持集和查询集的文件名
                                data_file.create_dataset('support_filenames', data=support_filenames,
                                                         dtype=h5.string_dtype())
                                data_file.create_dataset('query_filenames', data=[current_query_filename],
                                                         dtype=h5.string_dtype())

                                # 保存支持集和查询集的xyz最小值
                                data_file.create_dataset('support_xyz_mins', data=support_xyz_mins, dtype='float32')
                                data_file.create_dataset('query_xyz_mins', data=current_query_xyz_min[np.newaxis, ...],
                                                         dtype='float32')

                            print('\t{0} saved!'.format(out_filename))
                            self.num_episode += 1

                            # 将生成的文件名添加到列表中
                            self.file_names.append(out_filename)
                            episode_ind += 1

        else:
            # 考虑启用 更便于 可视化的命名
            print('Valid dataset (%s) does not exist...\n Constructing...' % test_data_path)
            os.mkdir(test_data_path)
            class_comb = list(combinations(self.classes, n_way))  # [(),(),(),...]
            self.num_episode = len(class_comb) * num_episode_per_comb
            episode_ind = 0
            self.file_names = []
            # 每100个h5文件来自于一个dataset episode
            for sampled_classes in class_comb:
                sampled_classes = list(sampled_classes)
                for i in range(num_episode_per_comb):
                    # 获取支持集和查询集的点云、标签和文件名
                    data = dataset.__getitem__(
                        episode_ind, sampled_classes)
                    out_filename = os.path.join(test_data_path, '%d.h5' % episode_ind)
                    # 保存支持集和查询集数据
                    write_episode(out_filename, data)
                    # 将生成的文件名添加到列表中
                    self.file_names.append(out_filename)
                    episode_ind += 1
            self.num_episode = len(self.file_names)

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index):
        file_name = self.file_names[index]
        return read_episode(file_name) #返回从 *.h5 读取到的


def batch_test_task_collate(batch):
    batch_support_ptclouds, batch_support_masks, batch_query_ptclouds, batch_query_labels, \
    batch_support_filenames, batch_query_filenames, \
    batch_support_xyz_mins, batch_query_xyz_mins, batch_sampled_classes = batch[0]


    # 将点云数据转换为 torch tensor 并做转置处理
    data = [
        torch.from_numpy(batch_support_ptclouds).transpose(2, 3),  # 支持集点云数据
        torch.from_numpy(batch_support_masks),                     # 支持集掩码
        torch.from_numpy(batch_query_ptclouds).transpose(1, 2),    # 查询集点云数据
        torch.from_numpy(batch_query_labels.astype(np.int64)),     # 查询集标签
        batch_support_filenames,                                  # 支持集文件名
        batch_query_filenames,                                    # 查询集文件名
        torch.from_numpy(batch_support_xyz_mins),                 # 支持集 xyz_min
        torch.from_numpy(batch_query_xyz_mins)                    # 查询集 xyz_min
    ]

    return data, batch_sampled_classes

# 修改测试，h5写入点云的文件来源。每个h5文件包含一个episode的 支持数据 与 查询数据
def write_episode(out_filename, data):
    support_ptclouds, support_masks, query_ptclouds, query_labels, support_filenames, query_filenames, support_xyz_mins, query_xyz_mins, sampled_classes = data
    with h5.File(out_filename, 'w') as data_file:
        # 写入支持集和查询集的点云、掩膜、标签和类
        data_file.create_dataset('support_ptclouds', data=support_ptclouds, dtype='float32')
        data_file.create_dataset('support_masks', data=support_masks, dtype='int32')
        data_file.create_dataset('query_ptclouds', data=query_ptclouds, dtype='float32')
        data_file.create_dataset('query_labels', data=query_labels, dtype='int64')
        data_file.create_dataset('sampled_classes', data=sampled_classes, dtype='int32')

        # 保存支持集和查询集的文件名
        data_file.create_dataset('support_filenames', data=support_filenames, dtype=h5.string_dtype())
        data_file.create_dataset('query_filenames', data=query_filenames, dtype=h5.string_dtype())

        # 保存支持集和查询集的xyz最小值
        data_file.create_dataset('support_xyz_mins', data=support_xyz_mins, dtype='float32')
        data_file.create_dataset('query_xyz_mins', data=query_xyz_mins, dtype='float32')

    print('\t {0} saved! | classes: {1}'.format(out_filename, sampled_classes))


def read_episode(file_name):
    with h5.File(file_name, 'r') as data_file:
        # 读取支持集和查询集的点云、掩膜、标签和类
        support_ptclouds = data_file['support_ptclouds'][:]
        support_masks = data_file['support_masks'][:]
        query_ptclouds = data_file['query_ptclouds'][:]
        query_labels = data_file['query_labels'][:]
        sampled_classes = data_file['sampled_classes'][:]

        # 读取支持集和查询集的文件名
        support_filenames = data_file['support_filenames'][:]
        query_filenames = data_file['query_filenames'][:]

        # 读取支持集和查询集的xyz最小值
        support_xyz_mins = data_file['support_xyz_mins'][:]
        query_xyz_mins = data_file['query_xyz_mins'][:]

    return (
        support_ptclouds, support_masks, query_ptclouds, query_labels,
        support_filenames, query_filenames,
        support_xyz_mins, query_xyz_mins, sampled_classes)



################################################  Pre-train Dataset ################################################
# classes：train_classes 类别数字，class2scans 训练集字典
class MyPretrainDataset(Dataset):
    def __init__(self, data_path, classes, class2scans, mode='train', num_point=4096, pc_attribs='xyz',
                       pc_augm=False, pc_augm_config=None):
        super(MyPretrainDataset).__init__()
        self.data_path = data_path
        self.classes = classes # 训练集合数字
        self.num_point = num_point
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        train_block_names = []
        all_block_names = []
        for k, v in sorted(class2scans.items()):
            all_block_names.extend(v) # all列表，用以收集所有类别 k 的所有扫描块列表 v
            n_blocks = len(v)
            n_test_blocks = int(n_blocks * 0.1)
            n_train_blocks = n_blocks - n_test_blocks
            train_block_names.extend(v[:n_train_blocks])  #train_block_names 这个列表存储着每个类别划分后的训练块名称列表

        if mode == 'train':
            self.block_names = list(set(train_block_names))
        elif mode == 'test':
            self.block_names = list(set(all_block_names) - set(train_block_names))
        else:
            raise NotImplementedError('Mode is unknown!')

        print('[Pretrain Dataset] Mode: {0} | Num_blocks: {1}'.format(mode, len(self.block_names)))

    def __len__(self):
        return len(self.block_names)

    def __getitem__(self, index):
        block_name = self.block_names[index]
        # 这里的block name是经过index的，所以输入的是一个场景，这里的sample_pointcloud对每个场景采样2048个点，返回采样后的数据和label，其中label是指示各个点是否属于目标类别sampled_class(默认为0)
        ptcloud, label, _ = sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                           self.pc_augm_config, block_name, self.classes, random_sample=True)

        return torch.from_numpy(ptcloud.transpose().astype(np.float32)), torch.from_numpy(label.astype(np.int64))