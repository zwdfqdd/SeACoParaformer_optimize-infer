"""
ParaformerSANMDecoder + DecoderLayerSANM（从 FunASR 训练源码抽取）

特点：
- 主 decoder：16 层（decoders）+ 1 层（decoders3，仅 FFN）
- SeACo decoder：6 层（decoders）+ 1 层（decoders3，仅 FFN）
- self_attn = MultiHeadedAttentionSANMDecoder（仅 FSMN）
- src_attn = MultiHeadedAttentionCrossAtt（标准 cross-attention）
"""

from typing import Tuple

import torch
import torch.nn as nn

from .layers import LayerNorm, PositionwiseFeedForwardDecoderSANM
from .attention import MultiHeadedAttentionSANMDecoder, MultiHeadedAttentionCrossAtt
from .utils import sequence_mask, repeat


class DecoderLayerSANM(nn.Module):
    """单层 Paraformer SANM Decoder（FFN + self_attn + src_attn）。

    forward 顺序：tgt → norm1 → FFN → norm2 → self_attn → norm3 → src_attn
    （与 FunASR 训练版本一致，FFN 在最前）
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_rate):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        if self_attn is not None:
            self.norm2 = LayerNorm(size)
        if src_attn is not None:
            self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn:
            tgt = self.norm2(tgt)
            x, _ = self.self_attn(tgt, tgt_mask)
            x = residual + self.dropout(x)

        if self.src_attn is not None:
            residual = x
            x = self.norm3(x)
            x_src_attn = self.src_attn(x, memory, memory_mask, ret_attn=False)
            x = residual + self.dropout(x_src_attn)

        return x, tgt_mask, memory, memory_mask, cache

    def get_attn_mat(self, tgt, tgt_mask, memory, memory_mask=None):
        """获取 cross-attention 权重矩阵（用于 ASF 热词过滤）。"""
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)
        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x, _ = self.self_attn(tgt, tgt_mask)
            x = residual + x
        residual = x
        x = self.norm3(x)
        x_src_attn, attn_mat = self.src_attn(x, memory, memory_mask, ret_attn=True)
        return attn_mat


class ParaformerSANMDecoder(nn.Module):
    """Paraformer SANM Decoder。

    支持两种用法：
        1. 主 decoder：input_layer='embed', use_output_layer=True, att_layer_num=16, num_blocks=16
        2. SeACo decoder：wo_input_layer=True, use_output_layer=False, att_layer_num=6, num_blocks=6
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 16,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        use_output_layer: bool = True,
        wo_input_layer: bool = False,
        att_layer_num: int = 16,
        kernel_size: int = 21,
        sanm_shfit: int = 0,
        **kwargs,
    ):
        super().__init__()
        attention_dim = encoder_output_size

        # 输入 embedding
        if wo_input_layer:
            self.embed = None
        else:
            self.embed = nn.Sequential(nn.Embedding(vocab_size, attention_dim))

        self.after_norm = LayerNorm(attention_dim)

        if use_output_layer:
            self.output_layer = nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self.att_layer_num = att_layer_num
        self.num_blocks = num_blocks
        if sanm_shfit is None:
            sanm_shfit = (kernel_size - 1) // 2

        # decoders（含 self_attn + src_attn）
        self.decoders = repeat(
            att_layer_num,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                MultiHeadedAttentionSANMDecoder(
                    attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=sanm_shfit
                ),
                MultiHeadedAttentionCrossAtt(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
            ),
        )

        # decoders3（1 层，仅 FFN）
        self.decoders3 = repeat(
            1,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                None, None,
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
            ),
        )

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        return_hidden: bool = False,
        return_both: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = ys_in_pad
        # 推理时 batch=1，不需要 mask（attention 内部跳过 masked_fill）
        if self.training:
            tgt_mask = sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]
            memory_mask = sequence_mask(hlens, device=hs_pad.device)[:, None, :]
            if tgt_mask.size(1) != memory_mask.size(1):
                memory_mask = torch.cat((memory_mask, memory_mask[:, -2:-1, :]), dim=1)
        else:
            tgt_mask = None
            memory_mask = None

        x = tgt
        x, tgt_mask, memory, memory_mask, _ = self.decoders(x, tgt_mask, hs_pad, memory_mask)
        x, tgt_mask, memory, memory_mask, _ = self.decoders3(x, tgt_mask, memory, memory_mask)

        hidden = self.after_norm(x)

        # olens：训练时从 mask 算，推理时直接用 ys_in_lens
        if tgt_mask is not None:
            olens = tgt_mask.sum(1)
        else:
            olens = ys_in_lens

        if self.output_layer is not None and return_hidden is False:
            x = self.output_layer(hidden)
            return x, olens
        if return_both:
            x = self.output_layer(hidden)
            return x, hidden, olens
        return hidden, olens

    def forward_asf6(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ):
        """ASF 注意力分数过滤（用第 6 层的 cross-attention 权重）。"""
        tgt = ys_in_pad
        tgt_mask = sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]
        memory = hs_pad
        memory_mask = sequence_mask(hlens, device=memory.device)[:, None, :]

        # 走前 5 层
        for i in range(min(5, self.att_layer_num)):
            tgt, tgt_mask, memory, memory_mask, _ = self.decoders[i](
                tgt, tgt_mask, memory, memory_mask
            )
        # 第 6 层取 attn_mat
        if self.att_layer_num > 5:
            attn_mat = self.decoders[5].get_attn_mat(tgt, tgt_mask, memory, memory_mask)
        else:
            # SeACo decoder 层数 < 6 时用最后一层
            attn_mat = self.decoders[-1].get_attn_mat(tgt, tgt_mask, memory, memory_mask)
        return attn_mat
