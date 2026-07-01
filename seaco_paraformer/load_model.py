"""
模型加载：加载 SeACo-Paraformer 权重到 SeacoParaformer。

从**本地预打包目录**加载（默认 ./models/asr/pt，不触发任何下载）。
不依赖 funasr / modelscope 运行时。
"""

import os
import torch
import logging
from pathlib import Path

from .model import SeacoParaformer

logger = logging.getLogger(__name__)

# 默认本地 PT 模型目录（含 model.pt / am.mvn / tokens.json / seg_dict）
DEFAULT_MODEL_DIR = os.getenv("PT_MODEL_DIR", "./models/asr/pt")


# 默认配置（对应 speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404）
DEFAULT_CONFIG = {
    "vocab_size": 8404,
    "inner_dim": 512,
    "bias_encoder_bid": False,
    "seaco_weight": 1.0,
    "NO_BIAS": 8377,
    "sos": 1,
    "encoder_conf": {
        "input_size": 560,
        "output_size": 512,
        "attention_heads": 4,
        "linear_units": 2048,
        "num_blocks": 50,
        "dropout_rate": 0.0,
        "positional_dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
        "input_layer": "pe",
        "kernel_size": 11,
        "sanm_shfit": 0,
    },
    "decoder_conf": {
        "attention_heads": 4,
        "linear_units": 2048,
        "num_blocks": 16,
        "dropout_rate": 0.0,
        "positional_dropout_rate": 0.0,
        "self_attention_dropout_rate": 0.0,
        "src_attention_dropout_rate": 0.0,
        "att_layer_num": 16,
        "kernel_size": 11,
        "sanm_shfit": 0,
    },
    "predictor_conf": {
        "idim": 512,
        "threshold": 1.0,
        "l_order": 1,
        "r_order": 1,
        "tail_threshold": 0.45,
        "smooth_factor": 1.0,
        "noise_threshold": 0,
        "smooth_factor2": 0.25,
        "noise_threshold2": 0.01,
        "upsample_times": 3,
    },
    "seaco_decoder_conf": {
        "attention_heads": 4,
        "linear_units": 1024,
        "num_blocks": 6,
        "att_layer_num": 6,
        "dropout_rate": 0.0,
        "positional_dropout_rate": 0.0,
        "self_attention_dropout_rate": 0.0,
        "src_attention_dropout_rate": 0.0,
        "kernel_size": 21,
        "sanm_shfit": 0,
    },
}


def _find_checkpoint(model_dir: str) -> str:
    """在模型目录中查找权重文件。"""
    for name in ["model.pt", "model.pth", "pytorch_model.bin"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    for f in Path(model_dir).rglob("*.pt"):
        return str(f)
    for f in Path(model_dir).rglob("*.pth"):
        return str(f)
    raise FileNotFoundError(f"未找到权重文件: {model_dir}")


def load_model(
    model_id: str = DEFAULT_MODEL_DIR,
    device: str = "cpu",
    cache_dir: str = None,
    config: dict = None,
) -> SeacoParaformer:
    """加载 SeACo-Paraformer 模型（仅本地目录，不联网下载）。

    Args:
        model_id: 本地模型目录路径（含权重文件），默认 ./models/asr/pt。
        device: 'cpu' 或 'cuda'
        cache_dir: 兼容保留，未使用
        config: 自定义配置，None 时使用 DEFAULT_CONFIG

    Returns:
        model: SeacoParaformer 实例（已加载权重，eval 模式）
    """
    cfg = config or DEFAULT_CONFIG

    # 仅支持本地预打包模型目录（不触发任何下载）
    if not os.path.isdir(model_id):
        raise FileNotFoundError(
            f"模型目录不存在: {model_id}\n"
            f"  请将 PT 权重放到本地目录（默认 ./models/asr/pt，含 model.pt），"
            f"或通过 --model-id / PT_MODEL_DIR 指定正确路径。本项目不联网下载模型。"
        )
    model_dir = model_id
    logger.info(f"使用本地模型目录: {model_dir}")

    ckpt_path = _find_checkpoint(model_dir)
    logger.info(f"权重文件: {ckpt_path}")

    # 创建模型
    model = SeacoParaformer(**cfg)

    # 加载权重
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"缺失的权重 ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"多余的权重 ({len(unexpected)}): {unexpected[:5]}...")

    model.eval()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型加载完成: {total_params / 1e6:.1f}M 参数")
    return model
