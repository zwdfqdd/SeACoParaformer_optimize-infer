"""
SANMEncoder + EncoderLayerSANM（从 FunASR 训练源码抽取）

特点：
- 50 层（encoders0=1 + encoders=49）
- 输入 560 维 → 输出 512 维（encoders0 兼做投影）
- 残差 Add 后可选 clamp（fp16 安全），通过 SANMEncoder(clamp_value=...) 控制
  - PT 推理：clamp_value=None（默认），保持原版数学等价
  - fp16/int8 ONNX 导出：60000（后段层残差激活峰值高达 ~48万 >> fp16 上限 65504，
    60000 贴近上限最大化保留信息；clamp=30000 裁剪过狠已弃用）
- 最终 after_norm
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .layers import LayerNorm, SinusoidalPositionEncoder, PositionwiseFeedForward
from .attention import MultiHeadedAttentionSANM
from .utils import repeat, sequence_mask


class EncoderLayerSANM(nn.Module):
    """单层 SANM Encoder（normalize_before=True，无 concat_after）。"""

    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        clamp_value: Optional[float] = None,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.clamp_value = clamp_value

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)

        if self.in_size == self.size:
            x = residual + self.dropout(self.self_attn(x, mask))
        else:
            x = self.dropout(self.self_attn(x, mask))
        if self.clamp_value is not None:
            x = x.clamp(min=-self.clamp_value, max=self.clamp_value)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if self.clamp_value is not None:
            x = x.clamp(min=-self.clamp_value, max=self.clamp_value)
        return x, mask


class SANMEncoder(nn.Module):
    """SANM Encoder（Paraformer 50 层）。

    Args:
        clamp_value: 残差 Add 后的 clamp 阈值。
            - None（默认）：不 clamp，PT 推理与原版数学等价
            - 60000：fp16/int8 ONNX 导出推荐值（后段层残差激活峰值高达 ~48万，
                     远超 fp16 上限 65504；60000 贴近上限最大化保留信息，仅极少数峰值点被裁）
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 512,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 50,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "pe",
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        clamp_value: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        self.clamp_value = clamp_value

        # 输入层（仅支持 pe）
        if input_layer != "pe":
            raise ValueError(f"unsupported input_layer: {input_layer}")
        self.embed = SinusoidalPositionEncoder()

        # FFN args
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate)

        # SANM attention args（第一层 in_size=input_size，后续 in_size=output_size）
        encoder_selfattn_layer_args0 = (
            attention_heads, input_size, output_size,
            attention_dropout_rate, kernel_size, sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads, output_size, output_size,
            attention_dropout_rate, kernel_size, sanm_shfit,
        )

        # encoders0: 1 层（input_size → output_size）
        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size, output_size,
                MultiHeadedAttentionSANM(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                clamp_value=clamp_value,
            ),
        )
        # encoders: num_blocks-1 层（output_size → output_size）
        self.encoders = repeat(
            num_blocks - 1,
            lambda lnum: EncoderLayerSANM(
                output_size, output_size,
                MultiHeadedAttentionSANM(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                clamp_value=clamp_value,
            ),
        )

        self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # 推理时 batch=1，无需 padding mask（attention 内部跳过 masked_fill）
        masks = None
        xs_pad = xs_pad * self.output_size() ** 0.5
        xs_pad = self.embed(xs_pad)

        xs_pad, masks = self.encoders0(xs_pad, masks)
        xs_pad, masks = self.encoders(xs_pad, masks)

        xs_pad = self.after_norm(xs_pad)

        # olens 直接用 ilens（推理时 batch=1 等同 seq_len），第三返回保持兼容
        return xs_pad, ilens, None
