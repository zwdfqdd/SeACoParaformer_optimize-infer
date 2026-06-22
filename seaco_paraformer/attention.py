"""
SANM Attention 模块（self-attention + FSMN）。

抽取自 FunASR 训练版本，去掉 LoRA、chunk、流式 mask 等运行时无关分支。
"""

import torch
import torch.nn as nn


class MultiHeadedAttentionSANM(nn.Module):
    """Encoder Self-Attention with SANM (FSMN memory)。"""

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(1, 2)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(1, 2)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(1, 2)
        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = -float("inf")
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, x, mask):
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask)
        return att_outs + fsmn_memory


class MultiHeadedAttentionSANMDecoder(nn.Module):
    """Decoder Self-Attention（仅 FSMN，无 q/k/v 投影）。"""

    def __init__(self, n_feat, dropout_rate, kernel_size, sanm_shfit=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.kernel_size = kernel_size

    def forward(self, inputs, mask, cache=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        if x.size(1) != inputs.size(1):
            inputs = inputs[:, -1, :]
        x = x + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x, cache


class MultiHeadedAttentionCrossAtt(nn.Module):
    """Cross-Attention（query 来自 decoder，key/value 来自 encoder）。"""

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        encoder_output_size=None,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k_v = nn.Linear(
            n_feat if encoder_output_size is None else encoder_output_size, n_feat * 2
        )
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, x, memory):
        b = x.size(0)
        q = self.linear_q(x)
        q_h = torch.reshape(q, (b, -1, self.h, self.d_k)).transpose(1, 2)
        k_v = self.linear_k_v(memory)
        k, v = torch.split(k_v, int(self.h * self.d_k), dim=-1)
        k_h = torch.reshape(k, (b, -1, self.h, self.d_k)).transpose(1, 2)
        v_h = torch.reshape(v, (b, -1, self.h, self.d_k)).transpose(1, 2)
        return q_h, k_h, v_h

    def forward_attention(self, value, scores, mask, ret_attn=False):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = -float("inf")
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        if ret_attn:
            return self.linear_out(x), attn
        return self.linear_out(x)

    def forward(self, x, memory, memory_mask, ret_attn=False):
        q_h, k_h, v_h = self.forward_qkv(x, memory)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        return self.forward_attention(v_h, scores, memory_mask, ret_attn=ret_attn)
