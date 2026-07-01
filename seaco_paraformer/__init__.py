"""
SeACo-Paraformer 模型代码框架（从 FunASR 训练源码抽取）

不依赖 FunASR 运行时，可直接用于：
1. PyTorch 推理（含热词）
2. ONNX 导出
3. TRT 转换前的精度调整

模块结构：
    utils.py      — make_pad_mask, sequence_mask, repeat, MultiSequential
    layers.py     — LayerNorm, SinusoidalPositionEncoder, FFN
    attention.py  — MultiHeadedAttentionSANM / Decoder / CrossAtt
    predictor.py  — CifPredictorV3 + cif / cif_v1_export
    encoder.py    — SANMEncoder + EncoderLayerSANM
    decoder.py    — ParaformerSANMDecoder + DecoderLayerSANM
    model.py      — SeacoParaformer（主模型，含 SeACo 热词推理）
    load_model.py — 从本地目录加载权重（默认 ./models/asr/pt，不联网下载）
"""

from .load_model import load_model

__all__ = ["load_model"]
