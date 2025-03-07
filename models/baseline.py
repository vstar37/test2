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
        if config.use_x3d_es:
            return features, es 
        elif config.use_global_feature:
            return features, global_feat 
        else:
            return features

    def load_point_cloud(self, block_path, pc_attribs='xyzrgbXYZ'):
        data = np.load(block_path)  # data.shape = (num, 7)
        N = data.shape[0]  # number of points in this scan
        sampled_point_inds = np.random.choice(np.arange(N), 2048, replace=(N < config.num_points))

        np.random.shuffle(sampled_point_inds)

        data =  data[sampled_point_inds]
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
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        row_col_info = current_block_name.split("_row")[-1]

        current_row = int(row_col_info.split("_col")[0])
        current_col = int(row_col_info.split("_col")[-1])


        concat_blocks = []
        xyz_min_list = []
        index_offset = 0
        index_mask = []

        for row_offset, col_offset in neighbors:
            neighbor_row = current_row + row_offset
            neighbor_col = current_col + col_offset

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

        global_pc = torch.cat(concat_blocks, dim=-1)

        xyz = global_pc[0:3, :]  
        xyz_min = torch.amin(xyz, dim=1, keepdim=True)
        XYZ = (xyz - xyz_min) / torch.amax(xyz - xyz_min, dim=1, keepdim=True)
        global_pc[6:9, :] = XYZ  
        return global_pc, torch.tensor(index_mask, dtype=torch.long).to(global_pc.device)

    def generate_global_pc(self, current_block_pc, current_block_xyzmin):

        grid = {
            1: 'top-left', 2: 'top', 3: 'top-right',
            4: 'left', 5: 'center', 6: 'right',
            7: 'bottom-left', 8: 'bottom', 9: 'bottom-right'
        }

        direction_to_grid = {
            'a': [4, 1, 2],  
            'b': [1, 2, 3],  
            'c': [2, 3, 6],  
            'd': [1, 4, 7],  
            'e': [3, 6, 9], 
            'f': [4, 7, 8], 
            'g': [6, 9, 8],
            'h': [7, 8, 9], 
        }

        global_pc = []
        index_mask = []

        grid_points = self.divide_block_into_grid(current_block_pc)

        for direction_idx, direction in enumerate(direction_to_grid):
            neighbor_blocks = [
                current_block_pc[:, grid_points[grid_idx]] for grid_idx in direction_to_grid[direction]
            ]

            inferred_points = self.diffusion_inference_for_direction(
                neighbor_blocks, direction
            )

            global_pc.append(inferred_points)
            index_mask.extend([0] * inferred_points.shape[1])  

            if direction_idx == 3:
                global_pc.append(current_block_pc)
                index_mask.extend([1] * current_block_pc.shape[1]) 

        global_pc = torch.cat(global_pc, dim=1)  # [in_channels, total_num_points]

        return global_pc, index_mask

    def divide_block_into_grid(self, current_block_pc):

        grid_points = {}

        x_coords = current_block_pc[0, :] 
        y_coords = current_block_pc[1, :] 

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

        combined_pc = torch.cat(neighbor_blocks, dim=1)  # [in_channels, total_neighbor_points]
        total_points = combined_pc.shape[1]

        if total_points >= num_sample:
            combined_pc = farthest_point_sampling(combined_pc, num_sample)
        else:
            inferred_points = torch.zeros(combined_pc.shape[0], num_sample//3, device=combined_pc.device)
            return inferred_points

        xyz = combined_pc[:3, :]  
        rgb = combined_pc[3:6, :]  

        group_idx_xyz, new_xyz, group_xyz = self.get_knn_groups(xyz)
        group_idx_rgb, new_rgb, group_rgb = self.get_knn_groups(rgb)
        features = self.compute_clusters(group_xyz, group_rgb)
        xyz = features[:3, :]  
        rgb = features[3:6, :] 
        inferred_points = biased_lagrangian_interpolation_direction(xyz, rgb, direction, num_sample//3)

        return inferred_points

    def get_knn_groups(self, point_cloud, k=16):

        group_idx, new_xyz = self.knn_search(point_cloud, k=k)

        group_xyz = torch.gather(
            point_cloud.unsqueeze(1).expand(-1, k, -1),  # [num_points, k, 3]
            dim=0,
            index=group_idx.unsqueeze(-1).expand(-1, -1, 3)  #  [num_points, k, 3]
        )

        return group_idx, new_xyz, group_xyz

    def knn_search(self, point_cloud, k):

        point_cloud = point_cloud.T  
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
        mean_xyz = group_xyz.mean(dim=1)  # 形状为 [num_points, 3]
        mean_rgb = group_rgb.mean(dim=1)  # 形状为 [num_points, 3]
        features = torch.cat([mean_xyz, mean_rgb], dim=1)  # 形状为 [num_points, 6]

        return features

    def compute_geometric_properties(self, clusters):
    
        geometric_properties = []

        for cluster_xyz in clusters:  # [k, 3] 
            print(cluster_xyz.shape)
            cluster_centered = cluster_xyz - cluster_xyz.mean(dim=0, keepdim=True)  
            cov_matrix = cluster_centered.T @ cluster_centered / (cluster_xyz.shape[0] - 1)  # [3, 3]
            eigvals = torch.linalg.eigvalsh(cov_matrix)  
            eigvals = eigvals.sort(descending=True)[0]  

            linearity = (eigvals[0] - eigvals[1]) / eigvals[0]
            planarity = (eigvals[1] - eigvals[2]) / eigvals[0]
            scattering = eigvals[2] / eigvals[0]

            geometric_properties.append([linearity, planarity, scattering])

        return torch.tensor(geometric_properties, device=clusters.device)  # [num_clusters, 3]


    def getGlobalFeature(self, x, x_name, x_xyz_min):

        with torch.no_grad():
            global_pcs = []
            index_masks = []

            if isinstance(x_name, bytes):
                x_name = x_name.decode('utf-8')

            expanded_x_name = []
            for item in x_name:
                if isinstance(item, bytes):
                    item = item.decode('utf-8') 
                item_cleaned = re.sub(r"(support|query)_\d+_", "", item)
                item_cleaned = item_cleaned.replace("\n", " ")  
                item_cleaned = item_cleaned.strip("[]").replace("'", "")  

                names = item_cleaned.split()
                expanded_x_name.extend(names)  

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
                global_pcs.append(global_pc.squeeze(0))  
                index_masks.append(index_mask)

            global_pcs = torch.stack(global_pcs, dim=0)
            global_pcs = global_pcs.permute(0, 2, 1)  # [B, C, N] -> [B, N, C]

            global_pcs_XYZ = global_pcs[:, :, 6:9]
            global_pcs_xyz = global_pcs[:, :, :3]
            global_pcs_rgb = global_pcs[:, :, 3:6]

            es_XYZ = self.es_encoder(global_pcs_XYZ).permute(0, 2, 1) 
            es_xyz = self.es_encoder(global_pcs_xyz).permute(0, 2, 1)
            es_rgb = self.es_encoder(global_pcs_rgb).permute(0, 2, 1) 

            es = torch.cat((es_xyz,es_rgb,es_XYZ), dim=1)

            selected_es = [es[i, :, mask].unsqueeze(0) for i, mask in enumerate(index_masks)]
            selected_es = torch.cat(selected_es, dim=0)  # [batch_size, out_dim, num_points]

            if config.use_global_feature:
                global_pcs = global_pcs.permute(0, 2, 1)  #[B, N, C] -> [B, C, N]

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
        feat_level1, feat_level2 = self.local_encoder(x) 
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
        mask_fg = ~mask_bg  # 
        support_bg_feature = bg_features
        support_fg_feature = support_feat[mask_fg]  
        q_mask_bg = (query_y == 0)  # Background mask
        q_mask_fg = ~q_mask_bg 
        query_bg_features = query_feat[q_mask_bg]  # Extract background features
        query_fg_features = query_feat[q_mask_fg]  # Extract background features

        feature_dict = {
            "support_bg_features": support_bg_feature.clone().detach() if support_bg_feature.numel() > 0 else None,
            "support_fg_features": support_fg_feature.clone().detach() if support_fg_feature.numel() > 0 else None,
            "query_bg_features": query_bg_features.clone().detach() if query_bg_features.numel() > 0 else None,
            "query_fg_features": query_fg_features.clone().detach() if query_fg_features.numel() > 0 else None,
        }

        torch.save(feature_dict, "tsne_features.pth")
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

            support_q = self.denoise_q(support_es)  # [B, N_points, feat_dim]
            support_v = self.denoise_v(support_es)  # [B, N_points, feat_dim]

            query_q = self.denoise_q(query_es)  # [B, N_points, feat_dim]
            query_v = self.denoise_v(query_es)  # [B, N_points, feat_dim]

            support_attn = torch.einsum('bnd,bmd->bnm', support_q, support_es)  # [B, N_points, N_points]
            query_attn = torch.einsum('bnd,bmd->bnm', query_q, query_es)  # [B, N_points, N_points]

            support_attn = F.softmax(support_attn, dim=-1)  # [B, N_points, N_points]
            query_attn = F.softmax(query_attn, dim=-1)  # [B, N_points, N_points]

            support_correction = torch.sum(support_attn.unsqueeze(-1) * support_v.unsqueeze(1),
                                           dim=2)  # [B, N_points, feat_dim]
            query_correction = torch.sum(query_attn.unsqueeze(-1) * query_v.unsqueeze(1),
                                         dim=2)  # [B, N_points, feat_dim]
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

        prototype_quality_loss_1 = self.sup_regulize_Loss(feature_memory_1, support_feat, support_y)
        prototype_quality_loss_2 = self.sup_regulize_Loss(feature_memory_2, support_feat, support_y)

        ICDL_1 = self.inter_class_difference_loss(feature_memory_1)
        ICDL_2 = self.inter_class_difference_loss(feature_memory_2)

        loss_p1 = prototype_quality_loss_1 + ICDL_1
        loss_p2 = prototype_quality_loss_2 + ICDL_2

        probs = F.softmax(logits_1, dim=-1)  
        top2_vals, _ = probs.topk(2, dim=-1)
        confidence_scores_1 = top2_vals[..., 0] - top2_vals[..., 1] 

        low_confidence_mask = confidence_scores_1 < config.rectify_threshold  
        low_confidence_ratio = low_confidence_mask.sum().item() / (low_confidence_mask.numel() + 1e-6)

        w_1 = 1 / (loss_p1 + 1e-6)
        w_2 = 1 / (loss_p2 + 1e-6)
        ac_1 = w_1 / (w_1 + w_2)
        ac_2 = w_2 / (w_1 + w_2)

        '''
        log_file = "low_confidence_ratio_log.txt"  # 

        def log_low_confidence_ratio(ratio_1, ratio_final):
            with open(log_file, "a", buffering=1) as f: 
                f.write(f"{ratio_1:.6f},{ratio_final:.6f}\n") 
                f.flush()  
                os.fsync(f.fileno()) 
        '''
        # CBLR
        if config.logits_strategy == 0:
            if low_confidence_ratio < config.confidence_ratio_threshold:
                final_logits = logits_1
                print('strategy==0, p1 is good.')
            else:
                final_logits = ac_1 * logits_1 + ac_2 * logits_2

            probs_final = F.softmax(final_logits, dim=-1)
            top2_vals_final, _ = probs_final.topk(2, dim=-1)
            confidence_scores_final = top2_vals_final[..., 0] - top2_vals_final[..., 1]
            low_confidence_mask_final = confidence_scores_final < config.rectify_threshold
            #low_confidence_ratio_final = low_confidence_mask_final.sum().item() / (
            #            low_confidence_mask_final.numel() + 1e-6)

        elif config.logits_strategy == 1:
            final_logits = logits_1.clone()  
            final_logits[low_confidence_mask] = (
                    ac_1 * logits_1[low_confidence_mask] + ac_2 * logits_2[low_confidence_mask]
            )
            '''
            num_low_confidence = low_confidence_mask.sum().item()  
            num_high_confidence = low_confidence_mask.numel() - num_low_confidence  
            print(f"Number of high-confidence points: {num_high_confidence}")
            print(f"Number of low-confidence points: {num_low_confidence}")
            '''
        else:
            final_logits = ac_1 * logits_1 + ac_2 * logits_2

        # ADAL
        alpha = 1 / (1 + torch.exp(-0.3 * (loss_p1 - loss_p2)))

        cross_entropy_loss = F.cross_entropy(final_logits.reshape(-1, self.n_way + 1), query_y.reshape(-1).long())
        loss_p1_weighted = (1 - alpha) * loss_p1
        loss_p2_weighted = alpha * loss_p2

        loss = [cross_entropy_loss, loss_p1_weighted, loss_p2_weighted]
        '''
        loss_log_file = "loss_log.txt" 

        def log_loss(cross_entropy_loss, loss_p1_weighted, loss_p2_weighted):
            total_loss = cross_entropy_loss + loss_p1_weighted + loss_p2_weighted

            if total_loss > 1e-6:
                cross_entropy_loss /= total_loss
                loss_p1_weighted /= total_loss
                loss_p2_weighted /= total_loss

            with open(loss_log_file, "a", buffering=1) as f: 
                f.write(f"{cross_entropy_loss:.6f}, {loss_p1_weighted:.6f}, {loss_p2_weighted:.6f}\n")
                f.flush()  
                os.fsync(f.fileno())  
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

        batch_size, num_channels, num_points = support_features.shape
        support_features = support_features.flatten(start_dim=1)  # (batch_size, num_channels * num_points)
        corr_support = F.cosine_similarity(support_features, prototypes.unsqueeze(0).expand(batch_size, -1), dim=1)  # (batch_size, num_classes)

        query_features = query_features.flatten(start_dim=1)  # (batch_size, num_channels * num_points)
        corr_query = F.cosine_similarity(query_features, prototypes.unsqueeze(0).expand(batch_size, -1), dim=1)  # (batch_size, num_classes)

        loss_support = torch.mean(torch.abs(corr_support - 1)) 
        loss_query = torch.mean(torch.abs(corr_query - 1))  

        loss = (loss_support + loss_query) / 2
        return loss
    def inter_class_difference_loss(self, prototypes):
        
        batch_size, class_num, feature_dim = prototypes.shape
）
        prototypes_flattened = prototypes.view(batch_size, class_num, -1)  # (batch_size, class_num, feature_dim)
        dist_matrix = torch.cdist(prototypes_flattened, prototypes_flattened,
                                  p=2)  # shape: (batch_size, class_num, class_num)

        similarity_matrix = torch.exp(-dist_matrix) 
        difference_matrix = 1 - similarity_matrix  

        loss = torch.zeros(batch_size, device=prototypes.device)

        for i in range(class_num):
            for j in range(i + 1, class_num):  
                # print(difference_matrix[:, i, j])
                mask = difference_matrix[:, i, j] < self.diff_threshold  
                if mask.any(): 
                    loss += torch.sum(torch.exp(self.diff_threshold - difference_matrix[:, i, j]) * mask)  

        return loss.mean()  


