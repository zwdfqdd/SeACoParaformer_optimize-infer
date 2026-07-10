"""
模型产物准备编排（启动时按 MODEL_PRECISION 检查 + 逐级转换）

PT 权重需提前下载并打包进镜像（不在运行时下载）：
    默认目录 models/asr/pt/（含 model.pt 等权重），可用环境变量 PT_MODEL_DIR 覆盖。

依赖链（每个产物缺失时，按链路从上游开始构建）：

    PT 权重（本地预打包 PT_MODEL_DIR）
    ├── 整体 ONNX            export_onnx_whole.py          → models/asr/fp32/model.onnx
    │   └── ONNX int8        convert_onnx_int8_dynamic.py  → models/asr/int8/model.onnx
    └── 分段 ONNX            export_onnx_split.py          → models/asr/split/{encoder,cif,decoder,bias_encoder,timestamp}.onnx
        ├── TRT fp32/fp16    convert_trt.py                → models/asr/trt/{gpu}_{module}_{prec}.engine
        └── QDQ ONNX         export_{encoder,cif,decoder,bias}_qdq → models/asr/split/{module}_qdq.onnx
            └── TRT int8     convert_trt.py                → models/asr/trt/{gpu}_{module}_int8_qdq.engine

各 MODEL_PRECISION 需要的产物（4 段主 engine；ENABLE_WORD_TIMESTAMP=true 时额外构建
timestamp 第 5 段，仅 fp16/fp32）：
    pt                 → PT 权重（本地，无需转换）
    onnx_fp32          → 整体 fp32 ONNX（启用时间戳时另需分段 ONNX 含 timestamp）
    onnx_int8          → 整体 fp32 ONNX → int8 ONNX
    trt_fp32           → 分段 ONNX → 4 段 fp32 engine（+timestamp fp32）
    trt_fp16           → 分段 ONNX → 4 段 fp16 engine（+timestamp fp16）
    trt_int8           → 分段 ONNX + 4 段 QDQ ONNX → 4 段 int8 engine（+timestamp fp16）
    trt_int8_enc       → 分段 ONNX + encoder QDQ → encoder int8 + 其余 fp16 engine（★线上推荐）

用法：
    python scripts/prepare_model.py                      # 用环境变量 MODEL_PRECISION
    python scripts/prepare_model.py --precision trt_int8_enc
    python scripts/prepare_model.py --precision trt_fp16 --check-only   # 仅检查不构建
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings, TRT_PRECISION_PROFILES


MODEL_DIR = "./models"
ASR_DIR = os.path.join(MODEL_DIR, "asr")
SPLIT_DIR = os.path.join(ASR_DIR, "split")
FP32_DIR = os.path.join(ASR_DIR, "fp32")
INT8_DIR = os.path.join(ASR_DIR, "int8")
TRT_DIR = os.path.join(ASR_DIR, "trt")
VAD_DIR = os.path.join(MODEL_DIR, "vad")
VAD_MODEL = os.path.join(VAD_DIR, "silero_vad.onnx")

CALIB_DATA = os.getenv("CALIB_DATA", "./calib_data/audio_data")

# 本地预打包 PT 模型目录（不在运行时下载）。
# 镜像内提前放好权重，导出脚本用 --model-id 指向该目录。
PT_MODEL_DIR = os.getenv("PT_MODEL_DIR", os.path.join(ASR_DIR, "pt"))


# ============================================================
# 环境能力检测
# ============================================================
def _can_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


CAP_MODELSCOPE = _can_import("modelscope")
CAP_SEACO = _can_import("seaco_paraformer")
CAP_ORT = _can_import("onnxruntime")
CAP_TRT = _can_import("tensorrt")
CAP_MODELOPT = _can_import("modelopt")


def _run(cmd: list[str]) -> bool:
    """执行子进程命令，返回是否成功。"""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def _gpu_name() -> str:
    return Settings._get_gpu_name()


# ============================================================
# 产物存在性检查
# ============================================================
def has_onnx_fp32() -> bool:
    return os.path.exists(os.path.join(FP32_DIR, "model.onnx"))


def has_onnx_int8() -> bool:
    return os.path.exists(os.path.join(INT8_DIR, "model.onnx"))


def has_split_onnx(need_timestamp: bool = False) -> bool:
    """检查分段 ONNX 是否齐全（need_timestamp=True 时含第 5 段 timestamp）。"""
    modules = ["encoder", "cif", "decoder", "bias_encoder"]
    if need_timestamp:
        modules.append("timestamp")
    for m in modules:
        if not os.path.exists(os.path.join(SPLIT_DIR, f"{m}.onnx")):
            return False
    return True


def has_qdq_onnx(module: str) -> bool:
    return os.path.exists(os.path.join(SPLIT_DIR, f"{module}_qdq.onnx"))


def has_trt_engine(module: str, precision: str) -> bool:
    """复用 config 的 engine 查找逻辑。"""
    return Settings._find_trt_engine(precision, module) is not None


# ============================================================
# 构建步骤（缺失则调用对应脚本）
# ============================================================
def ensure_split_onnx() -> bool:
    """确保 4 段普通 split ONNX 存在（缺失则导出）。"""
    if has_split_onnx():
        print("[OK] 分段 ONNX 已存在")
        return True
    if not CAP_SEACO:
        print("[缺失] 分段 ONNX 不存在，且当前环境无 seaco_paraformer，无法导出")
        return False
    if not ensure_pt():
        return False
    print("[构建] 导出分段 ONNX...")
    return _run([sys.executable, "scripts/export_onnx_split.py",
                 "--output-dir", SPLIT_DIR, "--clamp-value", "60000",
                 *_model_id_args()])


def ensure_qdq_onnx(module: str) -> bool:
    """确保某段 QDQ ONNX 存在（缺失则用 modelopt 量化导出）。"""
    if has_qdq_onnx(module):
        print(f"[OK] {module} QDQ ONNX 已存在")
        return True
    if not CAP_MODELOPT:
        print(f"[缺失] {module} QDQ ONNX 不存在，且当前环境无 nvidia-modelopt，无法量化导出")
        return False

    if module == "encoder":
        if not ensure_pt():
            return False
        print("[构建] 导出 encoder QDQ ONNX...")
        return _run([sys.executable, "scripts/export_encoder_qdq.py",
                     "--calib-data", CALIB_DATA,
                     "--output", os.path.join(SPLIT_DIR, "encoder_qdq.onnx"),
                     *_model_id_args(), *_cmvn_args()])
    elif module == "cif":
        # cif QDQ 需要 fp16 encoder engine 生成校准输入
        gpu = _gpu_name()
        enc_engine = os.path.join(TRT_DIR, f"{gpu}_encoder_fp16.engine")
        if not os.path.exists(enc_engine):
            print(f"[依赖] cif QDQ 需要 fp16 encoder engine，先构建它")
            if not ensure_trt_engine("encoder", "fp16"):
                return False
        if not ensure_pt():
            return False
        print("[构建] 导出 cif QDQ ONNX...")
        return _run([sys.executable, "scripts/export_cif_qdq.py",
                     "--calib-data", CALIB_DATA,
                     "--encoder-engine", enc_engine,
                     "--output", os.path.join(SPLIT_DIR, "cif_qdq.onnx"),
                     *_model_id_args(), *_cmvn_args()])
    elif module == "decoder":
        # decoder QDQ 需要 fp16 encoder/cif engine 生成校准输入
        gpu = _gpu_name()
        enc_engine = os.path.join(TRT_DIR, f"{gpu}_encoder_fp16.engine")
        cif_engine = os.path.join(TRT_DIR, f"{gpu}_cif_fp16.engine")
        if not (os.path.exists(enc_engine) and os.path.exists(cif_engine)):
            print(f"[依赖] decoder QDQ 需要 fp16 encoder/cif engine，先构建它们")
            if not ensure_trt_engine("encoder", "fp16"):
                return False
            if not ensure_trt_engine("cif", "fp16"):
                return False
        if not ensure_pt():
            return False
        print("[构建] 导出 decoder QDQ ONNX...")
        return _run([sys.executable, "scripts/export_decoder_qdq.py",
                     "--encoder-engine", enc_engine,
                     "--cif-engine", cif_engine,
                     "--output", os.path.join(SPLIT_DIR, "decoder_qdq.onnx"),
                     *_model_id_args(), *_cmvn_args()])
    elif module == "bias_encoder":
        # bias QDQ 自包含（无需上游 engine），用词表编码 token 校准
        if not ensure_pt():
            return False
        print("[构建] 导出 bias_encoder QDQ ONNX...")
        return _run([sys.executable, "scripts/export_bias_qdq.py",
                     "--output", os.path.join(SPLIT_DIR, "bias_encoder_qdq.onnx"),
                     *_model_id_args(), *_tokens_args()])
    else:
        print(f"[跳过] {module} 无 QDQ 导出脚本")
        return False


def ensure_trt_engine(module: str, precision: str) -> bool:
    """确保某段 TRT engine 存在（缺失则构建）。"""
    if has_trt_engine(module, precision):
        print(f"[OK] {module} {precision} engine 已存在")
        return True
    if not CAP_TRT:
        print(f"[缺失] {module} {precision} engine 不存在，且当前环境无 tensorrt")
        return False

    gpu = _gpu_name()
    profile = "bias" if module == "bias_encoder" else module

    if precision == "int8":
        # int8 走 QDQ ONNX
        if not ensure_qdq_onnx(module):
            return False
        onnx = os.path.join(SPLIT_DIR, f"{module}_qdq.onnx")
        engine = os.path.join(TRT_DIR, f"{gpu}_{module}_int8_qdq.engine")
    else:
        if not ensure_split_onnx():
            return False
        onnx = os.path.join(SPLIT_DIR, f"{module}.onnx")
        engine = os.path.join(TRT_DIR, f"{gpu}_{module}_{precision}.engine")

    print(f"[构建] {module} {precision} engine...")
    return _run([sys.executable, "scripts/convert_trt.py",
                 "--input", onnx, "--precision", precision,
                 "--profile", profile, "--output", engine])


def ensure_pt() -> bool:
    """确保本地 PT 权重目录存在（不在运行时下载，需提前打包进镜像）。"""
    ckpt_found = False
    if os.path.isdir(PT_MODEL_DIR):
        for name in ("model.pt", "model.pth", "pytorch_model.bin"):
            if os.path.exists(os.path.join(PT_MODEL_DIR, name)):
                ckpt_found = True
                break
        if not ckpt_found:
            # 递归找 .pt/.pth
            for ext in ("*.pt", "*.pth"):
                if list(Path(PT_MODEL_DIR).rglob(ext)):
                    ckpt_found = True
                    break
    if ckpt_found:
        print(f"[OK] 本地 PT 权重目录就绪: {PT_MODEL_DIR}")
        return True
    # 缺失 → 告警并自动下载（download_asr.py 从 ModelScope 拉取到 PT_MODEL_DIR）
    print(f"[警告] 本地 PT 权重缺失: {PT_MODEL_DIR}，自动启动下载脚本...")
    ok = _run([sys.executable, "scripts/download_asr.py", "--output-dir", PT_MODEL_DIR])
    if ok and os.path.isdir(PT_MODEL_DIR):
        for name in ("model.pt", "model.pth", "pytorch_model.bin"):
            if os.path.exists(os.path.join(PT_MODEL_DIR, name)):
                print(f"[OK] PT 权重下载完成: {PT_MODEL_DIR}")
                return True
        for ext in ("*.pt", "*.pth"):
            if list(Path(PT_MODEL_DIR).rglob(ext)):
                print(f"[OK] PT 权重下载完成: {PT_MODEL_DIR}")
                return True
    print(f"[失败] PT 权重下载失败，请手动运行 scripts/download_asr.py 或设 PT_MODEL_DIR")
    return False


def _model_id_args() -> list[str]:
    """若本地 PT 目录存在，返回 --model-id 参数，否则返回空（用脚本默认在线 ID）。"""
    if os.path.isdir(PT_MODEL_DIR):
        return ["--model-id", PT_MODEL_DIR]
    return []


def _cmvn_args() -> list[str]:
    """显式传 cmvn 路径（配置文件在 PT_MODEL_DIR=models/asr/pt 下）。"""
    cmvn_path = os.path.join(PT_MODEL_DIR, "am.mvn")
    if os.path.exists(cmvn_path):
        return ["--cmvn-path", cmvn_path]
    return []


def _tokens_args() -> list[str]:
    """显式传 tokens 路径（bias QDQ 用，配置文件在 PT_MODEL_DIR 下）。"""
    tokens_path = os.path.join(PT_MODEL_DIR, "tokens.json")
    if os.path.exists(tokens_path):
        return ["--tokens-path", tokens_path]
    return []


def ensure_onnx_fp32() -> bool:
    if has_onnx_fp32():
        print("[OK] 整体 fp32 ONNX 已存在")
        return True
    if not CAP_SEACO:
        print("[缺失] 整体 fp32 ONNX 不存在，且无 seaco_paraformer，无法导出")
        return False
    if not ensure_pt():
        return False
    print("[构建] 导出整体 fp32 ONNX...")
    return _run([sys.executable, "scripts/export_onnx_whole.py",
                 "--output-dir", ASR_DIR, "--skip-fp16", *_model_id_args()])


def ensure_onnx_int8() -> bool:
    if has_onnx_int8():
        print("[OK] 整体 int8 ONNX 已存在")
        return True
    if not ensure_onnx_fp32():
        return False
    if not CAP_ORT:
        print("[缺失] 无 onnxruntime，无法动态量化 int8")
        return False
    print("[构建] 整体 fp32 → int8 动态量化...")
    return _run([sys.executable, "scripts/convert_onnx_int8_dynamic.py",
                 "--input-dir", FP32_DIR, "--output-dir", INT8_DIR])


def ensure_vad() -> bool:
    """确保 Silero VAD 模型存在（服务启动必需）。缺失则尝试下载。"""
    if os.path.exists(VAD_MODEL):
        print("[OK] VAD 模型已存在")
        return True
    print(f"[构建] VAD 模型缺失，尝试下载到 {VAD_DIR} ...")
    ok = _run([sys.executable, "scripts/download_vad.py", "--output-dir", VAD_DIR])
    if not ok:
        print(f"[警告] VAD 模型下载失败，请手动放置: {VAD_MODEL}")
    return ok


PUNC_DIR = os.path.join(MODEL_DIR, "punc")


def _has_punc_model() -> bool:
    """判断 models/punc 下扁平结构标点模型是否齐全（prune*.bin + vocab.json + merges.txt）。"""
    if not os.path.isdir(PUNC_DIR):
        return False
    order = Settings.PUNC_NGRAM_ORDER
    prune = os.path.join(PUNC_DIR, f"prune{''.join(map(str, range(order)))}.bin")
    vocab = os.path.join(PUNC_DIR, "vocab.json")
    merges = os.path.join(PUNC_DIR, "merges.txt")
    return os.path.exists(prune) and os.path.exists(vocab) and os.path.exists(merges)


def ensure_punc() -> bool:
    """确保 ngram 标点模型存在（仅 ENABLE_SENTENCE_TIMESTAMP 启用时需要）。缺失则下载。

    句子级时间戳强依赖字级时间戳；此处仅负责标点模型产物就绪，开关校验在服务侧。
    缺失仅告警不阻断（句子级会降级回段级）。
    """
    if _has_punc_model():
        print("[OK] ngram 标点模型已存在")
        return True
    order = Settings.PUNC_NGRAM_ORDER
    print(f"[构建] ngram 标点模型缺失，尝试下载到 {PUNC_DIR}（order={order}）...")
    ok = _run([sys.executable, "scripts/download_punc.py",
               "--output-dir", PUNC_DIR, "--order", str(order),
               "--tokenizer-id", Settings.PUNC_TOKENIZER_ID])
    if not ok:
        print(f"[警告] 标点模型下载失败，句子级时间戳将降级回段级: {PUNC_DIR}")
    return ok


# ============================================================
# 按 precision 编排
# ============================================================
def prepare(precision: str, check_only: bool = False) -> bool:
    """按精度准备所有需要的产物。返回是否全部就绪。"""
    print(f"\n>>> 准备模型产物: MODEL_PRECISION={precision}")
    print(f"环境能力: modelscope={CAP_MODELSCOPE} seaco={CAP_SEACO} "
          f"ort={CAP_ORT} trt={CAP_TRT} modelopt={CAP_MODELOPT}\n")

    if check_only:
        return _check_only(precision)

    # VAD 模型服务启动必需，所有精度都先确保（缺失则下载，失败仅告警不阻断）
    ensure_vad()

    # 句子级时间戳标点模型（仅启用时准备，缺失下载失败仅降级不阻断）
    if Settings.ENABLE_SENTENCE_TIMESTAMP:
        ensure_punc()

    if precision == "pt":
        return ensure_pt()

    if precision == "onnx_fp32":
        ok = ensure_onnx_fp32()
        # 启用字级时间戳时，ORT 走分段串联，需额外确保分段 ONNX（含 timestamp）存在
        if Settings.ENABLE_WORD_TIMESTAMP:
            if not has_split_onnx(need_timestamp=True):
                print("[准备] 字级时间戳启用，ORT 分段串联需分段 ONNX（含 timestamp）")
                ensure_split_onnx()
        return ok

    if precision == "onnx_int8":
        return ensure_onnx_int8()

    if precision in TRT_PRECISION_PROFILES:
        prec_map = TRT_PRECISION_PROFILES[precision]
        ok = True
        # encoder/cif/decoder 为核心段，必须就绪
        for module in ("encoder", "cif", "decoder"):
            if not ensure_trt_engine(module, prec_map[module]):
                ok = False
        # bias_encoder 缺失仅影响热词，不阻断
        if not ensure_trt_engine("bias_encoder", prec_map["bias_encoder"]):
            print("[警告] bias_encoder engine 缺失，热词功能不可用")
        # timestamp 第 5 段：仅 ENABLE_WORD_TIMESTAMP 开启时构建。
        # 精度由 prec_map 决定（trt_fp32→fp32，其余→fp16；BLSTM 不量化，int8 已在
        # config 兜底回退 fp16）。缺失仅影响字级时间戳，不阻断主链路。
        if Settings.ENABLE_WORD_TIMESTAMP:
            ts_prec = Settings.get_trt_precision_map().get("timestamp", "fp16")
            if not ensure_trt_engine("timestamp", ts_prec):
                print("[警告] timestamp engine 缺失，字级时间戳不可用（words 为空）")
        return ok

    print(f"[错误] 未知精度: {precision}")
    return False


def _check_only(precision: str) -> bool:
    """仅检查产物是否齐全，不构建。"""
    if precision == "pt":
        ok = ensure_pt()
        print(f"pt（本地权重 {PT_MODEL_DIR}）: {'OK' if ok else '缺失'}")
        return ok
    if precision == "onnx_fp32":
        ok = has_onnx_fp32(); print(f"onnx_fp32: {'OK' if ok else '缺失'}"); return ok
    if precision == "onnx_int8":
        ok = has_onnx_int8(); print(f"onnx_int8: {'OK' if ok else '缺失'}"); return ok
    if precision in TRT_PRECISION_PROFILES:
        prec_map = TRT_PRECISION_PROFILES[precision]
        ok = True
        for module in ("encoder", "cif", "decoder", "bias_encoder"):
            exists = has_trt_engine(module, prec_map[module])
            print(f"{module} {prec_map[module]}: {'OK' if exists else '缺失'}")
            if module != "bias_encoder" and not exists:
                ok = False
        return ok
    return False


def main():
    parser = argparse.ArgumentParser(description="模型产物准备编排")
    parser.add_argument("--precision", default=None,
                        help="目标精度（默认读环境变量 MODEL_PRECISION）")
    parser.add_argument("--check-only", action="store_true", help="仅检查，不构建")
    args = parser.parse_args()

    # 解析精度（含 auto 探测）
    if args.precision:
        Settings.MODEL_PRECISION = args.precision
    precision = Settings.get_model_precision()

    print("=" * 60)
    print("SeACo-Paraformer 模型产物准备")
    print("=" * 60)

    ok = prepare(precision, check_only=args.check_only)

    print("\n" + "=" * 60)
    if ok:
        print(f"[成功] {precision} 所需产物已就绪")
        sys.exit(0)
    else:
        print(f"[失败] {precision} 产物不齐全，请检查上方日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
