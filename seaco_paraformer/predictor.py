"""
CIF Predictor V3：从 encoder hidden 预测 token 数量并生成 acoustic embeddings。

抽取自 FunASR 训练版本，包含原始 cif 和向量化 cif_v1（用于 ONNX 导出）。
默认 cif 使用 for 循环（PT 推理用），向量化版本用于 ONNX 导出。
"""

import torch
import torch.nn as nn

from .layers import LayerNorm
from .utils import make_pad_mask


def cif(hidden, alphas, threshold):
    """原始 CIF 实现（含 for 循环，PT 推理用）。

    输出 acoustic_embeds 按 batch 内最大 token 数 pad。
    """
    batch_size, len_time, hidden_size = hidden.size()
    integrate = torch.zeros([batch_size], device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
    list_fires = []
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size], device=hidden.device) - integrate
        integrate += alpha
        list_fires.append(integrate)
        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place, integrate - torch.ones([batch_size], device=hidden.device), integrate
        )
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur
        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(
            fire_place[:, None].repeat(1, hidden_size),
            remainds[:, None] * hidden[:, t, :],
            frame,
        )

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(batch_size):
        fire = fires[b, :]
        l = torch.index_select(
            frames[b, :, :], 0, torch.nonzero(fire >= threshold).squeeze(-1)
        )
        pad_l = torch.zeros(
            [max_label_len - l.size(0), hidden_size], device=hidden.device
        )
        list_ls.append(torch.cat([l, pad_l], 0))
    return torch.stack(list_ls, 0), fires


def cif_v1_export(hidden: torch.Tensor, alphas: torch.Tensor, threshold: float) -> tuple:
    """向量化 CIF（无 for 循环，用于 ONNX 导出，TRT 兼容）。

    使用 cumsum + one-hot + bmm，避免 ScatterElements 不被 TRT 支持的问题。
    """
    b, t, d = hidden.size()
    cum_alphas = torch.cumsum(alphas, dim=1)
    floor_cum = torch.floor(cum_alphas / threshold)

    # token 数量
    token_num = floor_cum[:, -1].long()
    # max_token_num 用输入长度 t 作为上界，避免 .item() 固化
    max_token_num = t

    # peak 标记
    floor_diff = torch.zeros_like(floor_cum)
    floor_diff[:, 1:] = floor_cum[:, 1:] - floor_cum[:, :-1]
    floor_diff[:, 0] = floor_cum[:, 0]
    cif_peak = (floor_diff > 0).float()

    # token assignment
    token_idx = (floor_cum - 1).long().clamp(min=0, max=max_token_num - 1)

    # one-hot assignment matrix
    assign = torch.zeros(b, t, max_token_num, device=hidden.device, dtype=hidden.dtype)
    assign.scatter_(2, token_idx.unsqueeze(-1), 1.0)

    weighted_assign = assign * alphas.unsqueeze(-1)
    acoustic_embeds = torch.bmm(weighted_assign.transpose(1, 2), hidden)

    return acoustic_embeds, cif_peak


class CifPredictorV3(nn.Module):
    """CIF Predictor V3（含上采样 cnn_blstm 时间戳预测）。

    配置固定：
        upsample_type = cnn_blstm
        use_cif1_cnn = False
        tail_threshold = 0.45
    """

    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0,
        tail_threshold=0.0,
        smooth_factor2=1.0,
        noise_threshold2=0,
        upsample_times=5,
        **kwargs,
    ):
        super().__init__()
        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.upsample_times = upsample_times

        # cnn_blstm 上采样
        self.upsample_cnn = nn.ConvTranspose1d(idim, idim, upsample_times, upsample_times)
        self.blstm = nn.LSTM(
            idim, idim, 1, bias=True, batch_first=True, dropout=0.0, bidirectional=True
        )
        self.cif_output2 = nn.Linear(idim * 2, 1)

        self.smooth_factor2 = smooth_factor2
        self.noise_threshold2 = noise_threshold2

    def forward(self, hidden, target_label=None, mask=None, ignore_id=-1,
                mask_chunk_predictor=None, target_label_length=None):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))

        # alphas2 (timestamp head)：use_cif1_cnn=False → 用 context
        _output = context
        output2 = self.upsample_cnn(_output)
        output2 = output2.transpose(1, 2)
        output2, (_, _) = self.blstm(output2)
        alphas2 = torch.sigmoid(self.cif_output2(output2))
        alphas2 = torch.nn.functional.relu(alphas2 * self.smooth_factor2 - self.noise_threshold2)

        if mask is not None:
            mask2 = (
                mask.repeat(1, self.upsample_times, 1)
                .transpose(-1, -2)
                .reshape(alphas2.shape[0], -1)
            )
            mask2 = mask2.unsqueeze(-1)
            alphas2 = alphas2 * mask2
        alphas2 = alphas2.squeeze(-1)
        token_num2 = alphas2.sum(-1)

        # alphas (main head)
        output = output.transpose(1, 2)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)

        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        if mask is not None:
            mask = mask.squeeze(-1)

        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None

        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, token_num, mask=mask)

        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak, token_num2

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            alphas = torch.cat([alphas, tail_threshold], dim=1)

        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)
        return hidden, alphas, token_num_floor
