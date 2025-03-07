""" Cuda util function

tweak
"""
import numpy as np
import torch


def cast_cuda(input):
    if isinstance(input, list):  # 递归处理列表
        for i in range(len(input)):
            input[i] = cast_cuda(input[i])
    elif isinstance(input, np.ndarray):
        # 检查 ndarray 是否为 object 类型
        if input.dtype == np.object_:
            return input  # 跳过 object 类型 (字符串或混合类型)
        else:
            return torch.tensor(input).cuda()
    elif isinstance(input, torch.Tensor):  # 如果是 Tensor
        return input.cuda()
    return input  # 跳过其他类型