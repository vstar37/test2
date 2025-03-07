""" Util functions for loading and saving checkpoints

Author: Zhao Na, 2020
"""
import os
import torch

# 从预训练字典中加载 encoder 权重
def load_pretrain_checkpoint(model, pretrain_checkpoint_path):
    # Load pretrained model for point cloud encoding
    model_dict = model.state_dict()
    if pretrain_checkpoint_path is not None:
        pretrained_checkpoint = torch.load(os.path.join(pretrain_checkpoint_path, 'checkpoint.tar'))
        pretrained_dict = pretrained_checkpoint['params']

        # Add prefix for encoder keys
        updated_pretrained_dict = {'encoder.local_encoder.' + k: v for k, v in pretrained_dict.items()}

        # Initialize counters for matched and unmatched keys
        matched_keys = 0
        unmatched_keys = 0
        filtered_dict = {}

        # Compare and filter keys
        for k, v in updated_pretrained_dict.items():
            if k in model_dict and model_dict[k].size() == v.size():
                filtered_dict[k] = v
                matched_keys += 1
            else:
                unmatched_keys += 1

        # Update model weights
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

        print(f"Number of matched weights: {matched_keys}")
        print(f"Number of unmatched weights: {unmatched_keys}")
    else:
        raise ValueError('Pretrained checkpoint must be given.')

    return model

# 加载模型
def load_model_checkpoint(model_checkpoint_path, mode='test'):
    try:
        checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.pt'))
        start_iter = checkpoint['iteration']
        start_iou = checkpoint['IoU']
    except:
        raise ValueError('Model checkpoint file must be correctly given (%s).' %model_checkpoint_path)

    if mode == 'test':
        print('Load model checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return checkpoint['model']
    else:
        return checkpoint['model'], checkpoint['optimizer'], start_iter, start_iou

def save_pretrain_checkpoint(model, output_path):
    torch.save(dict(params=model.encoder.state_dict()), os.path.join(output_path, 'checkpoint.tar'))