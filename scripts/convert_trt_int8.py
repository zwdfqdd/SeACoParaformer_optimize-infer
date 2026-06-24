"""
ONNX → TensorRT INT8 engine（Entropy Calibrator）

支持 encoder / decoder 两个段（cif/bias 保持 fp16，体积小且数值敏感）。

策略：
- INT8 + fp16 fallback：TRT 自动决定哪些算子用 INT8 / fp16
- IInt8EntropyCalibrator2：基于 KL 散度的熵校准
- 校准数据从音频文件夹读取，复用项目内的 feature_extractor

用法：
    # encoder INT8（其他段保持 fp16）
    python scripts/convert_trt_int8.py \\
        --input ./models/asr/split/encoder.onnx \\
        --profile encoder \\
        --calib-data ./speech \\
        --calib-cache ./models/asr/trt/cache_encoder_int8.cache

    # decoder INT8（先用 fp16 encoder/cif 跑出 acoustic_embeds）
    python scripts/convert_trt_int8.py \\
        --input ./models/asr/split/decoder.onnx \\
        --profile decoder \\
        --calib-data ./speech \\
        --calib-cache ./models/asr/trt/cache_decoder_int8.cache \\
        --encoder-engine ./models/asr/trt/2080_ti_encoder_fp16.engine \\
        --cif-engine ./models/asr/trt/2080_ti_cif_fp16.engine

校准数据要求：
- ./speech 目录下放 16kHz 单声道 WAV 音频
- 推荐 200-500 条，覆盖 2s/4s/8s 各时长（与 bucket 边界对齐）
- 不需要标签，仅用于收集 activation 分布
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import tensorrt as trt
except ImportError:
    sys.exit("需要 tensorrt: pip install tensorrt")

try:
    import torch
except ImportError:
    sys.exit("需要 torch")

try:
    import soundfile as sf
except ImportError:
    sys.exit("需要 soundfile: pip install soundfile")


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ============================================================
# Profile 配置（与 convert_trt.py 对齐）
# ============================================================
ENCODER_PROFILE = {
    "speech": {"min": (1, 8, 560), "opt": (1, 128, 560), "max": (8, 289, 560)},
}

DECODER_PROFILE = {
    "acoustic_embeds": {"min": (1, 2, 512), "opt": (1, 128, 512), "max": (8, 289, 512)},
    "encoder_out": {"min": (1, 8, 512), "opt": (1, 128, 512), "max": (8, 289, 512)},
    "bias_embed": {"min": (1, 1, 512), "opt": (1, 4, 512), "max": (8, 32, 512)},
}

PROFILES = {
    "encoder": ENCODER_PROFILE,
    "decoder": DECODER_PROFILE,
}


# ============================================================
# Bucket 对齐（与 src/scheduler.py 一致）
# ============================================================
BUCKET_SEQ_LENS = [34, 67, 134]  # 2s/4s/8s 对应 LFR 帧数


def get_bucket_seq_len(seq_len: int) -> int:
    """将 seq_len pad 到最近的 bucket 边界。"""
    for b in BUCKET_SEQ_LENS:
        if seq_len <= b:
            return b
    return BUCKET_SEQ_LENS[-1]


# ============================================================
# Encoder Calibrator
# ============================================================
class EncoderCalibrator(trt.IInt8EntropyCalibrator2):
    """encoder 校准器：从音频文件读取 → 特征提取 → 喂给 TRT 校准。

    重要：所有校准 batch 必须 pad 到统一固定 shape（calib_seq_len），
    否则 TRT 无法稳定收集每个 tensor 的 activation 直方图，
    会导致大量 tensor "Missing scale" 并 fall back 到非 INT8（量化失效）。
    """

    def __init__(self, audio_dir: str, cache_path: str, batch_size: int = 1,
                 cmvn_path: str | None = None, calib_seq_len: int = 134):
        super().__init__()
        self._cache_path = cache_path
        self._batch_size = batch_size
        # 固定校准 shape（默认 134 = 8s 桶，覆盖最大长度）
        self._calib_seq_len = calib_seq_len

        # 收集音频文件
        self._audio_files = sorted([
            str(p) for p in Path(audio_dir).rglob("*.wav")
        ])
        if not self._audio_files:
            raise RuntimeError(f"未在 {audio_dir} 下找到 .wav 文件")
        print(f"  校准数据：{len(self._audio_files)} 条音频")
        print(f"  固定校准 shape: ({batch_size}, {calib_seq_len}, 560)")

        # 加载 CMVN
        from src.feature_extractor import extract_features, load_cmvn
        self._extract_features = extract_features
        if cmvn_path is None:
            cmvn_path = "./models/asr/am.mvn"
        self._cmvn_mean, self._cmvn_istd = load_cmvn(cmvn_path)

        # 进度
        self._idx = 0
        self._device_buffer = None  # GPU 内存

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: list) -> list | None:
        """返回下一批校准数据的 GPU 指针；None 表示结束。

        所有 batch 统一 pad/截断到固定 shape (_calib_seq_len)。
        """
        if self._idx >= len(self._audio_files):
            return None

        # 取下一条音频，预处理为特征
        audio_path = self._audio_files[self._idx]
        self._idx += 1

        try:
            pcm, sr = sf.read(audio_path, dtype="float32")
            if len(pcm.shape) > 1:
                pcm = pcm[:, 0]
            if sr != 16000:
                print(f"  跳过非 16kHz: {audio_path} ({sr}Hz)")
                return self.get_batch(names)

            features = self._extract_features(
                pcm, sample_rate=sr,
                cmvn_mean=self._cmvn_mean, cmvn_istd=self._cmvn_istd,
            )
            # 统一 pad/截断到固定校准 shape
            target_len = self._calib_seq_len
            padded = np.zeros((self._batch_size, target_len, 560), dtype=np.float32)
            valid = min(features.shape[0], target_len)
            padded[0, :valid, :] = features[:valid]

            # 上传到 GPU
            t = torch.from_numpy(padded).cuda().contiguous()
            self._device_buffer = t  # 持有引用，防止 GC
            print(f"  [{self._idx}/{len(self._audio_files)}] {Path(audio_path).name} "
                  f"({pcm.size/sr:.2f}s → {features.shape[0]} 帧 → 固定 {target_len})")
            return [t.data_ptr()]
        except Exception as e:
            print(f"  跳过 {audio_path}: {e}")
            return self.get_batch(names)

    def read_calibration_cache(self) -> bytes | None:
        if os.path.exists(self._cache_path):
            print(f"  使用现有 calibration cache: {self._cache_path}")
            with open(self._cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes):
        Path(self._cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "wb") as f:
            f.write(cache)
        print(f"  calibration cache 已写入: {self._cache_path}")


# ============================================================
# Decoder Calibrator（输入是上游 fp16 engine 跑出来的中间结果）
# ============================================================
class DecoderCalibrator(trt.IInt8EntropyCalibrator2):
    """decoder 校准器：用 fp16 encoder + cif 跑出 acoustic_embeds 喂给 decoder。"""

    def __init__(self, audio_dir: str, cache_path: str,
                 encoder_engine_path: str, cif_engine_path: str,
                 cmvn_path: str | None = None,
                 calib_enc_len: int = 134, calib_tok_len: int = 60):
        super().__init__()
        self._cache_path = cache_path
        # 固定校准 shape
        self._calib_enc_len = calib_enc_len  # encoder_out 序列长度
        self._calib_tok_len = calib_tok_len  # acoustic_embeds token 数

        self._audio_files = sorted([
            str(p) for p in Path(audio_dir).rglob("*.wav")
        ])
        if not self._audio_files:
            raise RuntimeError(f"未在 {audio_dir} 下找到 .wav 文件")
        print(f"  校准数据：{len(self._audio_files)} 条音频")
        print(f"  固定校准 shape: enc_len={calib_enc_len}, tok_len={calib_tok_len}")

        from src.feature_extractor import extract_features, load_cmvn
        self._extract_features = extract_features
        if cmvn_path is None:
            cmvn_path = "./models/asr/am.mvn"
        self._cmvn_mean, self._cmvn_istd = load_cmvn(cmvn_path)

        # 加载 encoder + cif
        from src.trt_engine import _TRTInferencer  # type: ignore
        self._encoder = _TRTInferencer(encoder_engine_path)
        self._cif = _TRTInferencer(cif_engine_path)
        print(f"  上游 engine 已加载: encoder={encoder_engine_path}, cif={cif_engine_path}")

        self._idx = 0
        self._gpu_buffers: dict[str, torch.Tensor] = {}

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names: list) -> list | None:
        if self._idx >= len(self._audio_files):
            return None

        audio_path = self._audio_files[self._idx]
        self._idx += 1

        try:
            pcm, sr = sf.read(audio_path, dtype="float32")
            if len(pcm.shape) > 1:
                pcm = pcm[:, 0]
            if sr != 16000:
                return self.get_batch(names)

            features = self._extract_features(
                pcm, sample_rate=sr,
                cmvn_mean=self._cmvn_mean, cmvn_istd=self._cmvn_istd,
            )
            # encoder/cif 用固定 enc_len 跑（与 encoder int8 校准 shape 对齐）
            enc_len = self._calib_enc_len
            padded = np.zeros((1, enc_len, 560), dtype=np.float32)
            valid = min(features.shape[0], enc_len)
            padded[0, :valid, :] = features[:valid]

            # 跑 encoder
            enc_inputs = {"speech": padded}
            if "speech_lengths" in self._encoder.input_names:
                enc_inputs["speech_lengths"] = np.array([enc_len], dtype=np.int64)
            enc_out = self._encoder.infer(enc_inputs)
            encoder_out = enc_out["encoder_out"]  # (1, enc_len, 512)

            # 跑 cif
            mask = np.ones((1, 1, enc_len), dtype=np.float32)
            cif_out = self._cif.infer({"encoder_out": encoder_out, "mask": mask})
            acoustic_embeds = cif_out["acoustic_embeds"]
            token_num = int(np.round(cif_out["token_num"].flatten()[0]))
            if token_num == 0:
                return self.get_batch(names)

            # acoustic_embeds 统一 pad/截断到固定 tok_len
            tok_len = self._calib_tok_len
            ae_fixed = np.zeros((1, tok_len, 512), dtype=np.float32)
            v = min(token_num, tok_len)
            ae_fixed[0, :v, :] = acoustic_embeds[0, :v, :]
            acoustic_embeds = ae_fixed.astype(np.float32)

            # bias_embed 全零（固定 1×1×512）
            bias_embed = np.zeros((1, 1, 512), dtype=np.float32)

            # 上传 decoder 输入
            ptrs = []
            for name in names:
                if name == "acoustic_embeds":
                    data = acoustic_embeds
                elif name == "encoder_out":
                    data = encoder_out.astype(np.float32)
                elif name == "bias_embed":
                    data = bias_embed
                elif name == "token_num":
                    data = np.array([tok_len], dtype=np.int64)
                elif name == "encoder_out_lens":
                    data = np.array([enc_len], dtype=np.int64)
                else:
                    raise RuntimeError(f"未知 decoder 输入: {name}")

                t = torch.from_numpy(np.ascontiguousarray(data)).cuda()
                self._gpu_buffers[name] = t  # 持有引用
                ptrs.append(t.data_ptr())

            print(f"  [{self._idx}/{len(self._audio_files)}] {Path(audio_path).name} "
                  f"→ token_num={token_num} (固定 {tok_len})")
            return ptrs
        except Exception as e:
            print(f"  跳过 {audio_path}: {e}")
            return self.get_batch(names)

    def read_calibration_cache(self) -> bytes | None:
        if os.path.exists(self._cache_path):
            print(f"  使用现有 calibration cache: {self._cache_path}")
            with open(self._cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes):
        Path(self._cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "wb") as f:
            f.write(cache)
        print(f"  calibration cache 已写入: {self._cache_path}")


# ============================================================
# 构建 INT8 engine
# ============================================================
def get_gpu_name() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()
        for p in ["nvidia ", "geforce ", "rtx ", "tesla "]:
            name = name.replace(p, "")
        return name.strip().replace(" ", "_")
    return "unknown_gpu"


def build_int8_engine(
    onnx_path: str,
    output_path: str,
    profile_type: str,
    calibrator: trt.IInt8Calibrator,
    workspace_gb: int = 4,
):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    print(f"\n[1/3] 解析 ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            sys.exit("ONNX 解析失败")
    print(f"  网络 layer 总数: {network.num_layers}")

    print(f"\n[2/3] 配置 builder（INT8 + fp16 fallback）...")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * 1024 * 1024 * 1024)

    # INT8 + fp16 fallback：让 TRT 在精度敏感处自动回退 fp16
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    # Dynamic shape profile
    if profile_type not in PROFILES:
        sys.exit(f"未知 profile: {profile_type}")
    profile = builder.create_optimization_profile()
    for name, p in PROFILES[profile_type].items():
        profile.set_shape(name, p["min"], p["opt"], p["max"])
    config.add_optimization_profile(profile)
    # INT8 校准也用同一 profile
    config.set_calibration_profile(profile)

    print(f"\n[3/3] 构建 engine（含 INT8 校准）...")
    print(f"  输出: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit("Engine 构建失败")
    build_time = time.perf_counter() - t0

    with open(output_path, "wb") as f:
        f.write(serialized)

    onnx_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    engine_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n构建完成！")
    print(f"  耗时: {build_time:.1f}s")
    print(f"  ONNX {onnx_mb:.1f}MB → Engine {engine_mb:.1f}MB")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ONNX → TRT INT8 engine（Entropy Calibrator）")
    parser.add_argument("--input", required=True, help="ONNX 模型路径")
    parser.add_argument("--output", default=None, help="输出 engine 路径（默认自动命名）")
    parser.add_argument("--profile", required=True, choices=["encoder", "decoder"],
                        help="量化对象（encoder 或 decoder，cif/bias 用 fp16 不需要 INT8）")
    parser.add_argument("--calib-data", default="./int8/calib_data/audio_data",
                        help="校准音频目录（递归扫描 *.wav，16kHz 单声道）")
    parser.add_argument("--calib-cache", default=None,
                        help="校准 cache 路径（默认 models/asr/trt/cache_{profile}_int8.cache）")
    parser.add_argument("--cmvn-path", default="./models/asr/am.mvn",
                        help="CMVN 参数文件")
    parser.add_argument("--workspace", type=int, default=4)

    # decoder 校准额外参数
    parser.add_argument("--encoder-engine", default=None,
                        help="（仅 decoder profile）上游 encoder fp16 engine 路径")
    parser.add_argument("--cif-engine", default=None,
                        help="（仅 decoder profile）上游 cif fp16 engine 路径")

    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"ONNX 不存在: {args.input}")
    if not Path(args.calib_data).is_dir():
        sys.exit(f"校准数据目录不存在: {args.calib_data}")

    # 自动输出路径
    if args.output is None:
        gpu_name = get_gpu_name()
        input_path = Path(args.input)
        model_name = input_path.stem
        output_dir = input_path.parent.parent / "trt"
        args.output = str(output_dir / f"{gpu_name}_{model_name}_int8.engine")

    if args.calib_cache is None:
        args.calib_cache = f"./models/asr/trt/cache_{args.profile}_int8.cache"

    print("=" * 60)
    print(f"ONNX → TRT INT8 engine（{args.profile}）")
    print(f"  ONNX:       {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Calib data: {args.calib_data}")
    print(f"  Calib cache: {args.calib_cache}")
    print(f"  GPU:        {get_gpu_name()}")
    print(f"  TRT:        {trt.__version__}")
    print("=" * 60)

    # 创建 calibrator
    if args.profile == "encoder":
        calibrator = EncoderCalibrator(
            audio_dir=args.calib_data,
            cache_path=args.calib_cache,
            cmvn_path=args.cmvn_path,
        )
    elif args.profile == "decoder":
        if not args.encoder_engine or not args.cif_engine:
            sys.exit("decoder 校准需要 --encoder-engine 和 --cif-engine")
        calibrator = DecoderCalibrator(
            audio_dir=args.calib_data,
            cache_path=args.calib_cache,
            encoder_engine_path=args.encoder_engine,
            cif_engine_path=args.cif_engine,
            cmvn_path=args.cmvn_path,
        )

    build_int8_engine(
        onnx_path=args.input,
        output_path=args.output,
        profile_type=args.profile,
        calibrator=calibrator,
        workspace_gb=args.workspace,
    )


if __name__ == "__main__":
    main()
