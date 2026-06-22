"""
工具函数：mask 生成、模块重复
"""

import torch
import torch.nn as nn


def make_pad_mask(lengths, maxlen=None):
    """生成 padding mask（True 表示 padding 位置）。

    使用纯 tensor 操作（不用 .tolist()），保证 ONNX 导出时 lengths 作为动态输入。

    Args:
        lengths: (B,) 每条序列的有效长度
        maxlen: 最大长度，None 则取 lengths.max()

    Returns:
        mask: (B, maxlen) bool tensor，True = padding 位置
    """
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths, dtype=torch.int64)

    bs = lengths.size(0)
    if maxlen is None:
        maxlen = int(lengths.max().item())

    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = lengths.unsqueeze(-1).to(torch.int64)
    mask = seq_range_expand >= seq_length_expand
    return mask


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    """生成 sequence mask（1 表示有效位置）。"""
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()
    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class MultiSequential(nn.Sequential):
    """支持多输入多输出的 Sequential。"""

    def forward(self, *args):
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """重复 N 次构造模块。"""
    return MultiSequential(*[fn(n) for n in range(N)])


def analyze_tensor_fp16_stats(tensor: torch.Tensor) -> dict:
    """
    统计 PyTorch 张量的最大最小值，以及超出 fp16 表示范围的数值个数。

    参数:
        tensor (torch.Tensor): 输入的张量。

    返回:
        dict: 包含统计结果的字典（仅在有溢出时返回详细信息）。
    """
    if tensor.numel() == 0:
        return {}

    max_val = tensor.max().item()
    min_val = tensor.min().item()
    fp16_max = torch.finfo(torch.float16).max

    count_greater = (tensor > fp16_max).sum().item()
    count_abs_greater = (tensor.abs() > fp16_max).sum().item()
    count_inf = torch.isinf(tensor).sum().item()

    if count_greater > 0 or count_abs_greater > 0:
        return {
            "max": max_val,
            "min": min_val,
            "fp16_max": fp16_max,
            "count_greater_than_fp16_max": count_greater,
            "count_abs_greater_than_fp16_max": count_abs_greater,
            "count_inf": count_inf,
        }
    else:
        return {}
