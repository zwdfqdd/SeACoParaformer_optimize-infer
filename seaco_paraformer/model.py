"""
SeacoParaformer 主模型（从 FunASR 训练源码抽取）

完整推理流程（含热词）：
    speech → encoder → predictor(CIF) → decoder → seaco_decoder(hotwords) → logits

子模块：
    encoder:              SANMEncoder (50 层)
    predictor:            CifPredictorV3
    decoder:              ParaformerSANMDecoder (16+1 层)
    seaco_decoder:        ParaformerSANMDecoder (6+1 层, wo_input_layer, no output_layer)
    bias_encoder:         LSTM (2 层, 热词 LSTM 编码)
    hotword_output_layer: Linear(512, vocab_size)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .encoder import SANMEncoder
from .decoder import ParaformerSANMDecoder
from .predictor import CifPredictorV3
from .utils import make_pad_mask


class SeacoParaformer(nn.Module):
    """SeACo-Paraformer 完整模型。"""

    def __init__(
        self,
        vocab_size: int = 8404,
        encoder_conf: dict = None,
        decoder_conf: dict = None,
        predictor_conf: dict = None,
        seaco_decoder_conf: dict = None,
        inner_dim: int = 512,
        bias_encoder_bid: bool = False,
        seaco_weight: float = 1.0,
        NO_BIAS: int = 8377,
        sos: int = 1,
        **kwargs,
    ):
        super().__init__()
        encoder_conf = encoder_conf or {}
        decoder_conf = decoder_conf or {}
        predictor_conf = predictor_conf or {}
        seaco_decoder_conf = seaco_decoder_conf or {}

        self.vocab_size = vocab_size
        self.inner_dim = inner_dim
        self.seaco_weight = seaco_weight
        self.NO_BIAS = NO_BIAS
        self.sos = sos

        # Encoder
        self.encoder = SANMEncoder(**encoder_conf)

        # Predictor
        self.predictor = CifPredictorV3(**predictor_conf)

        # Decoder（主 decoder）
        self.decoder = ParaformerSANMDecoder(
            vocab_size=vocab_size,
            encoder_output_size=inner_dim,
            **decoder_conf,
        )

        # SeACo decoder（6 层，无 input_layer，无 output_layer）
        self.seaco_decoder = ParaformerSANMDecoder(
            vocab_size=vocab_size,
            encoder_output_size=inner_dim,
            wo_input_layer=True,
            use_output_layer=False,
            **seaco_decoder_conf,
        )

        # Bias Encoder（热词 LSTM）
        self.bias_encoder = nn.LSTM(
            inner_dim, inner_dim, 2,
            batch_first=True,
            dropout=0.0,
            bidirectional=bias_encoder_bid,
        )
        if bias_encoder_bid:
            self.lstm_proj = nn.Linear(inner_dim * 2, inner_dim)
        else:
            self.lstm_proj = None

        # 热词输出层
        self.hotword_output_layer = nn.Linear(inner_dim, vocab_size)

    def encode(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        """encoder forward。返回 (encoder_out, encoder_out_lens)。"""
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        return encoder_out, encoder_out_lens

    def calc_predictor(self, encoder_out, encoder_out_lens):
        """CIF predictor forward，返回 (acoustic_embeds, token_num, alphas, cif_peak)。"""
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        outs = self.predictor(encoder_out, None, encoder_out_mask, ignore_id=-1)
        return outs[:4]

    def _hotword_representation(self, hotword_pad: torch.Tensor, hotword_lengths: torch.Tensor):
        """编码热词为 bias embedding。

        Args:
            hotword_pad: (H, L) — 热词 token IDs
            hotword_lengths: (H,) — 每个热词的实际长度

        Returns:
            selected: (H, D) — 每个热词的 embedding（取最后有效时间步）
        """
        # 使用 decoder 的 embedding 层
        hw_embed = self.decoder.embed(hotword_pad)  # (H, L, D)

        # LSTM 编码（pack_padded_sequence 处理变长）
        hw_embed_packed = nn.utils.rnn.pack_padded_sequence(
            hw_embed,
            hotword_lengths.cpu().long(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bias_encoder(hw_embed_packed)
        rnn_output = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)[0]

        if self.lstm_proj is not None:
            hw_hidden = self.lstm_proj(rnn_output)
        else:
            hw_hidden = rnn_output

        # 取每个热词最后一个有效时间步
        _ind = np.arange(0, hw_hidden.shape[0]).tolist()
        selected = hw_hidden[
            _ind,
            [i - 1 for i in hotword_lengths.detach().cpu().tolist()],
        ]
        return selected  # (H, D)

    def _seaco_decode_with_ASF(
        self,
        encoder_out,
        encoder_out_lens,
        sematic_embeds,
        ys_pad_lens,
        hw_list: List[List[int]] = None,
        nfilter: int = 50,
    ):
        """SeACo 完整推理（含 ASF 过滤 + NO_BIAS mask 合并）。

        Args:
            encoder_out: (B, T, D)
            encoder_out_lens: (B,)
            sematic_embeds: (B, N, D) — CIF 输出
            ys_pad_lens: (B,) — token 数量
            hw_list: 热词列表（每个元素是 token id 列表），末尾必须有 [sos] 占位
            nfilter: ASF 过滤后保留的 top-K 热词数

        Returns:
            decoder_out: (B, N, vocab) — log_softmax 后的 logits
        """
        seaco_weight = self.seaco_weight

        # 主 decoder（return_both 同时返回 logits 和 hidden）
        decoder_out, decoder_hidden, _ = self.decoder(
            encoder_out, encoder_out_lens,
            sematic_embeds, ys_pad_lens,
            return_hidden=True, return_both=True,
        )
        decoder_pred = torch.log_softmax(decoder_out, dim=-1)

        if hw_list is None:
            return decoder_pred

        # 编码热词
        hw_lengths = [len(i) for i in hw_list]
        hw_list_pad = self._pad_hotwords(hw_list)  # (H, L)
        hw_lengths_t = torch.tensor(hw_lengths, dtype=torch.long, device=encoder_out.device)
        selected = self._hotword_representation(hw_list_pad.to(encoder_out.device), hw_lengths_t)
        # selected: (H, D)

        # 扩展到 batch
        contextual_info = (
            selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
        )  # (B, H, D)
        num_hot_word = contextual_info.shape[1]
        _contextual_length = (
            torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)
        )

        # ASF 过滤（如果热词多于 nfilter）
        if nfilter > 0 and nfilter < num_hot_word:
            hotword_scores = self.seaco_decoder.forward_asf6(
                contextual_info, _contextual_length, decoder_hidden, ys_pad_lens
            )
            hotword_scores = hotword_scores[0].sum(0).sum(0)
            dec_filter = torch.topk(hotword_scores, min(nfilter, num_hot_word - 1))[1].tolist()
            add_filter = dec_filter
            add_filter.append(len(hw_list_pad) - 1)
            selected = selected[add_filter]
            contextual_info = (
                selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
            )
            num_hot_word = contextual_info.shape[1]
            _contextual_length = (
                torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)
            )

        # SeACo decoder × 2
        cif_attended, _ = self.seaco_decoder(
            contextual_info, _contextual_length, sematic_embeds, ys_pad_lens
        )
        dec_attended, _ = self.seaco_decoder(
            contextual_info, _contextual_length, decoder_hidden, ys_pad_lens
        )

        merged = cif_attended + dec_attended
        dha_output = self.hotword_output_layer(merged)
        dha_pred = torch.log_softmax(dha_output, dim=-1)

        # NO_BIAS mask 合并
        def _merge_res(dec_output, dha_output):
            lmbd = torch.Tensor([seaco_weight] * dha_output.shape[0]).to(dec_output.device)
            dha_ids = dha_output.max(-1)[1]
            dha_mask = (dha_ids == self.NO_BIAS).int().unsqueeze(-1)
            a = (1 - lmbd) / lmbd
            b = 1 / lmbd
            dha_mask = (dha_mask + a.reshape(-1, 1, 1)) / b.reshape(-1, 1, 1)
            logits = dec_output * dha_mask + dha_output * (1 - dha_mask)
            return logits

        merged_pred = _merge_res(decoder_pred, dha_pred)
        return merged_pred

    @staticmethod
    def _pad_hotwords(hw_list: List[List[int]]) -> torch.Tensor:
        """将热词列表 pad 为 tensor (H, max_len)。"""
        max_len = max(len(hw) for hw in hw_list)
        padded = torch.full((len(hw_list), max_len), 0, dtype=torch.long)
        for i, hw in enumerate(hw_list):
            padded[i, : len(hw)] = torch.tensor(hw, dtype=torch.long)
        return padded

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        """简化 forward（无热词版本）。返回 (logits, token_num)。"""
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        acoustic_embeds, token_num, alphas, cif_peak = self.calc_predictor(
            encoder_out, encoder_out_lens
        )
        pre_token_length = token_num.round().long()

        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens,
            acoustic_embeds, pre_token_length,
        )
        return decoder_out, token_num

    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        hw_list: Optional[List[List[int]]] = None,
        nfilter: int = 50,
    ):
        """完整推理（含热词）。

        Args:
            speech: (B, T, D)
            speech_lengths: (B,)
            hw_list: 热词列表（每个元素是 token id 列表）。末尾必须有 [sos] 占位。
            nfilter: ASF 过滤数

        Returns:
            logits: (B, N, vocab) — log_softmax 输出
            token_num: (B,)
        """
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        acoustic_embeds, token_num, alphas, cif_peak = self.calc_predictor(
            encoder_out, encoder_out_lens
        )
        pre_token_length = token_num.round().long()

        decoder_out = self._seaco_decode_with_ASF(
            encoder_out,
            encoder_out_lens,
            acoustic_embeds,
            pre_token_length,
            hw_list=hw_list,
            nfilter=nfilter,
        )
        return decoder_out, token_num
