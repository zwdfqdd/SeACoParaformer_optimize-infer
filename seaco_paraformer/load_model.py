"""
模型加载：从 ModelScope 下载 SeACo-Paraformer 权重并加载到 SeacoParaformer。

依赖：modelscope（仅下载权重文件）。不依赖 funasr 运行时。
"""

import os
import torch
import logging
from pathlib import Path

from .model import SeacoParaformer

logger = logging.getLogger(__name__)


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


def _download_model(model_id: str, cache_dir: str = None) -> str:
    """从 ModelScope 下载模型，返回本地目录。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            raise ImportError("需要 modelscope: pip install modelscope")
    return snapshot_download(model_id, cache_dir=cache_dir)


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
    model_id: str = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    device: str = "cpu",
    cache_dir: str = None,
    config: dict = None,
) -> SeacoParaformer:
    """加载 SeACo-Paraformer 模型。

    Args:
        model_id: ModelScope 模型 ID
        device: 'cpu' 或 'cuda'
        cache_dir: 模型缓存目录
        config: 自定义配置，None 时使用 DEFAULT_CONFIG

    Returns:
        model: SeacoParaformer 实例（已加载权重，eval 模式）
    """
    cfg = config or DEFAULT_CONFIG
    logger.info(f"下载模型: {model_id}")
    model_dir = _download_model(model_id, cache_dir)
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
