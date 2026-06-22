"""
基础层：LayerNorm, PositionEncoder, FFN
"""

import torch
import torch.nn as nn


class LayerNorm(torch.nn.LayerNorm):
    """LayerNorm 封装（eps=1e-12，与 FunASR 一致）。"""

    def __init__(self, nout):
        super().__init__(nout, eps=1e-12)


class SinusoidalPositionEncoder(nn.Module):
    """正弦位置编码（按 token 位置即时计算，无可学习参数）。"""

    def encode(self, positions: torch.Tensor, depth: int, dtype: torch.dtype = torch.float32):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype, device=device)
        ) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales, [1, 1, -1])
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)
        return x + position_encoding


class PositionwiseFeedForward(nn.Module):
    """Encoder 的 FFN（标准实现，与 FunASR 训练版本一致）。"""

    def __init__(self, idim, hidden_units, dropout_rate, activation=nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PositionwiseFeedForwardDecoderSANM(nn.Module):
    """Decoder 的 FFN（含 LayerNorm，与 FunASR 训练版本一致）。"""

    def __init__(self, idim, hidden_units, dropout_rate, activation=nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation
        self.norm = LayerNorm(hidden_units)

    def forward(self, x):
        return self.w_2(self.norm(self.dropout(self.activation(self.w_1(x)))))
