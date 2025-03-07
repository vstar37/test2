""" Data Preprocess and Loader for S3DIS Dataset

Author: Zhao Na, 2020
"""

import os
import glob
import numpy as np
import pickle


class S3DISDataset(object):
    def __init__(self, cvfold, data_path):
        self.data_path = data_path
        self.classes = 13
        # self.class2type = {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window', 6:'door', 7:'table',
        #                    8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
        class_names = open(os.path.join(os.path.dirname(data_path), 'meta', 's3dis_classnames.txt')).readlines()
        self.class2type = {i: name.strip() for i, name in enumerate(class_names)}
        print(self.class2type)
        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.types = self.type2class.keys()
        self.fold_0 = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'column']
        self.fold_1 = ['door', 'floor', 'sofa', 'table', 'wall', 'window']

        if cvfold == 0:
            self.test_classes = [self.type2class[i] for i in self.fold_0]
        elif cvfold == 1:
            self.test_classes = [self.type2class[i] for i in self.fold_1]
        else:
            raise NotImplementedError('Unknown cvfold (%s). [Options: 0,1]' %cvfold)

        all_classes = [i for i in range(0, self.classes-1)]
        self.train_classes = [c for c in all_classes if c not in self.test_classes]

        # print('train_class:{0}'.format(self.train_classes))
        # print('test_class:{0}'.format(self.test_classes))
        # self.class2scans 是一个字典，键是类别号(0-12), 值是列表，每个列表中都是所有包含该类别，并其符合阈值要求的场景块文件名(不包含后缀)
        self.class2scans = self.get_class2scans()
        self.class2scenes = self.get_class2scenes()


    def get_class2scenes(self):
        class2scenes = {}
        for class_id, scans in self.class2scans.items():
            scenes = set()  # 使用集合来去重
            for scan in scans:
                scene = self.extract_scene(scan)
                scenes.add(scene)
            class2scenes[class_id] = list(scenes)
        return class2scenes

    def extract_scene(self, scan):
        # 假设scene是scan名称的前部分 "Area_1_office_31_block_19_row1_col4" -> "Area_1_office_31"
        parts = scan.split('_')
        if len(parts) >= 4:
            scene = '_'.join(parts[:3])
        else:
            scene = scan  # 如果格式不对，则返回原始scan
        return scene

    def get_class2scans(self):
        class2scans_file = os.path.join(self.data_path, 'class2scans.pkl')
        if os.path.exists(class2scans_file):
            #load class2scans (dictionary)
            with open(class2scans_file, 'rb') as f:
                class2scans = pickle.load(f)
        else:
            min_ratio = 0.05  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k:[] for k in range(self.classes)} # 创建扫描字典
            # 遍历场景块
            for file in glob.glob(os.path.join(self.data_path, 'data', '*.npy')):
                scan_name = os.path.basename(file)[:-4]  # Area_1_conferenceRoom_1_block_0_row0_col0
                data = np.load(file)
                labels = data[:,6].astype(np.int64)
                classes = np.unique(labels) #这个块里包含的所有类别
                print('{0} | shape: {1} | classes: {2}'.format(scan_name, data.shape, list(classes)))
                # 检查场景中的每个类别情况
                for class_id in classes:
                    #if the number of points for the target class is too few, do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0]*min_ratio), min_pts)
                    # 阈值处理，确保这个块里的每个类都是有意义的
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name) # 往扫描字典对应的值列表中放入块名字

            print('==== class to scans mapping is done ====')
            for class_id in range(self.classes):
                print('\t class_id: {0} | min_ratio: {1} | min_pts: {2} | class_name: {3} | num of scans: {4}'.format(
                          class_id,  min_ratio, min_pts, self.class2type[class_id], len(class2scans[class_id])))

            with open(class2scans_file, 'wb') as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        return class2scans