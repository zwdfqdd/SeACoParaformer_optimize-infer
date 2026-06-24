"""
Decoder QDQ 量化导出（方案 1：Explicit Quantization）

与 encoder QDQ 同思路，但 decoder 输入有三个（acoustic_embeds + encoder_out + bias_embed），
校准数据需要先用 fp16 encoder + cif engine 跑出中间结果。

流程：
    1. 加载 PyTorch decoder（DecoderWithSeACoWrapper，独立包）
    2. 用 fp16 encoder/cif engine 对校准音频跑出 (acoustic_embeds, encoder_out)
    3. modelopt INT8 量化 + 校准（收集 amax）
    4. 导出含 QDQ 节点的 ONNX
    5. trtexec --int8 --fp16 构建（QDQ 自带 scale，不需要 calibrator）

用法：
    python scripts/export_decoder_qdq.py \\
        --encoder-engine ./models/asr/trt/2080_ti_encoder_fp16.engine \\
        --cif-engine ./models/asr/trt/2080_ti_cif_fp16.engine \\
        --output ./models/asr/split/decoder_qdq.onnx

导出后转 TRT：
    python scripts/convert_trt.py --input ./models/asr/split/decoder_qdq.onnx \\
        --precision int8 --profile decoder \\
        --output ./models/asr/trt/2080_ti_decoder_int8_qdq.engine
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _SimpleTRT:
    """自包含的单 engine TRT 推理器（不依赖 src.trt_engine）。"""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"engine 反序列化失败: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, inputs: dict) -> dict:
        d_inputs = {}
        for name in self.input_names:
            data = inputs[name]
            self.context.set_input_shape(name, data.shape)
            t = torch.from_numpy(data).cuda().contiguous()
            d_inputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        d_outputs = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            for i, s in enumerate(shape):
                if s <= 0:
                    shape[i] = list(inputs.values())[0].shape[0] if i == 0 else 300
            t = torch.zeros(shape, dtype=torch.float32, device="cuda")
            d_outputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        results = {}
        for name, t in d_outputs.items():
            actual = tuple(self.context.get_tensor_shape(name))
            if all(s > 0 for s in actual):
                slices = tuple(slice(0, s) for s in actual)
                results[name] = t[slices].cpu().numpy()
            else:
                results[name] = t.cpu().numpy()
        return results


def detect_backend(prefer: str | None = None) -> str:
    candidates = []
    if prefer:
        candidates.append(prefer)
    candidates += ["modelopt", "pytorch_quantization"]
    for name in candidates:
        if name == "modelopt":
            try:
                import modelopt.torch.quantization  # noqa
                return "modelopt"
            except ImportError:
                continue
        elif name == "pytorch_quantization":
            try:
                import pytorch_quantization  # noqa
                return "pytorch_quantization"
            except ImportError:
                continue
    sys.exit("未找到量化库，请安装 nvidia-modelopt==0.21.0 torchprofile")


def build_decoder_wrapper():
    """加载 DecoderWithSeACoWrapper。"""
    from seaco_paraformer.load_model import load_model
    from scripts.export_onnx_split import DecoderWithSeACoWrapper

    model = load_model()
    wrapper = DecoderWithSeACoWrapper(model)
    wrapper.eval()
    return wrapper


def collect_decoder_inputs(
    audio_dir: str, cmvn_path: str,
    encoder_engine: str, cif_engine: str,
    calib_enc_len: int = 134, calib_tok_len: int = 60,
    max_samples: int = 500,
) -> list[dict]:
    """用 fp16 encoder + cif 跑出 decoder 输入（统一固定 shape）。

    返回列表，每项是 {acoustic_embeds, token_num, encoder_out, encoder_out_lens, bias_embed} 的
    torch tensor 字典（CPU），供量化 forward_loop 使用。
    """
    import soundfile as sf
    from src.feature_extractor import extract_features, load_cmvn

    cmvn_mean, cmvn_istd = load_cmvn(cmvn_path)
    encoder = _SimpleTRT(encoder_engine)
    cif = _SimpleTRT(cif_engine)

    audio_files = sorted([str(p) for p in Path(audio_dir).rglob("*.wav")])[:max_samples]
    if not audio_files:
        sys.exit(f"未在 {audio_dir} 找到 .wav")

    samples = []
    for ap in audio_files:
        pcm, sr = sf.read(ap, dtype="float32")
        if len(pcm.shape) > 1:
            pcm = pcm[:, 0]
        if sr != 16000:
            continue

        features = extract_features(pcm, sample_rate=sr,
                                     cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
        enc_len = calib_enc_len
        padded = np.zeros((1, enc_len, 560), dtype=np.float32)
        valid = min(features.shape[0], enc_len)
        padded[0, :valid, :] = features[:valid]

        # encoder
        enc_inputs = {"speech": padded}
        if "speech_lengths" in encoder.input_names:
            enc_inputs["speech_lengths"] = np.array([enc_len], dtype=np.int64)
        enc_out = encoder.infer(enc_inputs)
        encoder_out = enc_out["encoder_out"]

        # cif
        mask = np.ones((1, 1, enc_len), dtype=np.float32)
        cif_out = cif.infer({"encoder_out": encoder_out, "mask": mask})
        acoustic_embeds = cif_out["acoustic_embeds"]
        token_num = int(np.round(cif_out["token_num"].flatten()[0]))
        if token_num == 0:
            continue

        # acoustic_embeds 统一固定 tok_len
        tok_len = calib_tok_len
        ae_fixed = np.zeros((1, tok_len, 512), dtype=np.float32)
        v = min(token_num, tok_len)
        ae_fixed[0, :v, :] = acoustic_embeds[0, :v, :]

        # bias_embed 全零（无热词校准；SeACo 路径在 wrapper 内自带计算）
        bias_embed = np.zeros((1, 1, 512), dtype=np.float32)

        samples.append({
            "acoustic_embeds": torch.from_numpy(ae_fixed).float(),
            "token_num": torch.tensor([tok_len], dtype=torch.long),
            "encoder_out": torch.from_numpy(encoder_out).float(),
            "encoder_out_lens": torch.tensor([enc_len], dtype=torch.long),
            "bias_embed": torch.from_numpy(bias_embed).float(),
        })

    print(f"  校准样本: {len(samples)} 条（enc_len={calib_enc_len}, tok_len={calib_tok_len}）")
    return samples


def quantize_with_modelopt(wrapper, samples, output_path, opset, calib_tok_len, calib_enc_len,
                           exclude_patterns=None):
    import modelopt.torch.quantization as mtq

    print("[modelopt] 配置 INT8 量化（INT8_DEFAULT_CFG）...")
    config = mtq.INT8_DEFAULT_CFG

    # 排除敏感模块（SeACo 热词路径等）不量化，保持 fp16
    if exclude_patterns:
        import copy
        config = copy.deepcopy(config)
        # modelopt config 的 quant_cfg 支持按模块名通配设置 enable=False
        quant_cfg = config.setdefault("quant_cfg", {})
        for pat in exclude_patterns:
            # 对匹配模块的 weight/input quantizer 关闭
            quant_cfg[f"*{pat}*"] = {"enable": False}
        print(f"  排除量化的模块模式: {exclude_patterns}")

    def forward_loop(model):
        with torch.no_grad():
            for i, s in enumerate(samples):
                model(
                    s["acoustic_embeds"], s["token_num"],
                    s["encoder_out"], s["encoder_out_lens"], s["bias_embed"],
                )
                if (i + 1) % 50 == 0:
                    print(f"  校准进度: {i + 1}/{len(samples)}")

    print("[modelopt] 量化 + 校准中...")
    mtq.quantize(wrapper, config, forward_loop)

    try:
        mtq.print_quant_summary(wrapper)
    except Exception as e:
        print(f"  （print_quant_summary 跳过: {e}）")

    print(f"\n[modelopt] 导出 QDQ ONNX: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # dummy 输入（与校准 shape 一致）
    dummy = (
        torch.randn(1, calib_tok_len, 512),                  # acoustic_embeds
        torch.tensor([calib_tok_len], dtype=torch.long),     # token_num
        torch.randn(1, calib_enc_len, 512),                  # encoder_out
        torch.tensor([calib_enc_len], dtype=torch.long),     # encoder_out_lens
        torch.randn(1, 3, 512),                              # bias_embed
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, output_path,
            opset_version=opset,
            input_names=["acoustic_embeds", "token_num", "encoder_out",
                         "encoder_out_lens", "bias_embed"],
            output_names=["logits"],
            dynamic_axes={
                "acoustic_embeds": {0: "batch", 1: "token_len"},
                "token_num": {0: "batch"},
                "encoder_out": {0: "batch", 1: "enc_len"},
                "encoder_out_lens": {0: "batch"},
                "bias_embed": {0: "batch", 1: "num_hotwords"},
                "logits": {0: "batch", 1: "logits_len"},
            },
            do_constant_folding=False,
        )

    # 检查 QDQ 节点
    try:
        import onnx
        m = onnx.load(output_path)
        q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
        dq = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
        print(f"\n  QDQ 节点统计: QuantizeLinear={q}, DequantizeLinear={dq}")
        if q == 0:
            print("  ⚠ 未发现 QuantizeLinear，QDQ 导出可能失败")
        else:
            print("  ✓ QDQ 节点已嵌入 ONNX")
    except Exception as e:
        print(f"  （ONNX QDQ 检查跳过: {e}）")


def main():
    parser = argparse.ArgumentParser(description="Decoder QDQ 量化导出（方案 1）")
    parser.add_argument("--calib-data", default="./int8/calib_data/audio_data")
    parser.add_argument("--cmvn-path", default="./models/asr/am.mvn")
    parser.add_argument("--encoder-engine", required=True, help="上游 encoder fp16 engine")
    parser.add_argument("--cif-engine", required=True, help="上游 cif fp16 engine")
    parser.add_argument("--output", default="./models/asr/split/decoder_qdq.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--calib-enc-len", type=int, default=134)
    parser.add_argument("--calib-tok-len", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--backend", default=None,
                        choices=["modelopt", "pytorch_quantization"])
    parser.add_argument("--exclude-patterns", nargs="*",
                        default=["seaco_decoder", "hotword_output_layer"],
                        help="排除量化的模块名模式（保持 fp16）。"
                             "默认排除 SeACo 热词路径（数值敏感，INT8 会破坏热词修正）。"
                             "传空 [] 则全部量化。")
    args = parser.parse_args()

    backend = detect_backend(args.backend)
    print("=" * 60)
    print("Decoder QDQ 量化导出（方案 1）")
    print(f"  量化库: {backend}")
    print(f"  校准数据: {args.calib_data}")
    print(f"  上游 encoder: {args.encoder_engine}")
    print(f"  上游 cif: {args.cif_engine}")
    print(f"  输出: {args.output}")
    print("=" * 60)

    print("\n[1/3] 加载 decoder...")
    wrapper = build_decoder_wrapper()

    print("\n[2/3] 用 fp16 encoder+cif 生成 decoder 校准输入...")
    samples = collect_decoder_inputs(
        args.calib_data, args.cmvn_path,
        args.encoder_engine, args.cif_engine,
        args.calib_enc_len, args.calib_tok_len, args.max_samples,
    )

    print("\n[3/3] 量化 + 导出...")
    if backend == "modelopt":
        quantize_with_modelopt(
            wrapper, samples, args.output, args.opset,
            args.calib_tok_len, args.calib_enc_len,
            exclude_patterns=args.exclude_patterns,
        )
    else:
        sys.exit("decoder QDQ 仅支持 modelopt 后端")

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\n完成！QDQ ONNX: {args.output} ({size_mb:.1f}MB)")
    print("\n下一步转 TRT：")
    print(f"  python scripts/convert_trt.py --input {args.output} \\")
    print(f"      --precision int8 --profile decoder \\")
    print(f"      --output ./models/asr/trt/2080_ti_decoder_int8_qdq.engine")


if __name__ == "__main__":
    main()
