# 技术栈

## 编程语言

- Python >= 3.12

## 核心框架与库

- **FunASR**: 阿里达摩院开源的端到端语音识别工具包，提供 Paraformer 模型基础架构（仅转换环境）
- **PyTorch + torchaudio**: 特征提取（kaldi fbank），不用于模型推理
- **ONNX Runtime GPU**: 模型推理引擎（v1 线上，fp32/int8）
- **TensorRT 10.6**: 高性能 GPU 推理引擎（v2 线上，fp16/INT8）
- **CUDA 12.6 + cuDNN 9**: GPU 计算基础
- **NumPy**: 数值计算
- **SoundFile**: 音频文件读取
- **FastAPI + Uvicorn**: HTTP 服务框架
- **Prometheus + OpenTelemetry**: 可观测性

## 推理引擎选择

| 场景 | 引擎 | 模型精度 | 说明 |
|------|------|----------|------|
| GPU 线上（v1） | ONNX Runtime | fp32 | 精度稳定，通用性好 |
| CPU 线上（v1） | ONNX Runtime | int8 | 动态量化，模型缩小 75% |
| GPU 线上（v2） | TensorRT | fp16/INT8 | 速度提升 2-3x，显存减半 |

## 构建与依赖管理

- `pip` + `requirements.txt` 管理 Python 依赖
- requirements-infer.txt：推理 + 现场转换统一依赖（转换推理合一镜像）
- Docker 单镜像：TRT 基础镜像内完成 PT→ONNX→engine 转换 + 推理

## 常用命令

```bash
# 安装推理依赖
pip install -r requirements-infer.txt

# 启动服务
python -m uvicorn src.main:app --host 0.0.0.0 --port 8080

# 运行测试
python tests/test_single.py --audio test_data/audio_16000_30s.wav

# 启动服务（容器内 entrypoint 自动按精度准备模型）
python scripts/prepare_model.py --precision trt_int8_enc
python -m uvicorn src.main:app --host 0.0.0.0 --port 8080

# 整体 ONNX 导出 + int8 动态量化（onnx_fp32 / onnx_int8）
python scripts/export_onnx_whole.py --skip-fp16 --output-dir ./models/asr
python scripts/convert_onnx_int8_dynamic.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

# 分段 ONNX 导出 + TRT 转换（trt 系列）
python scripts/export_onnx_split.py --output-dir ./models/asr/split --clamp-value 30000
python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --profile encoder
```

## 开发规范

- Python 文件名必须使用英文命名
- 文档及日志输出统一使用中文
- 功能代码完成后不自动生成测试文件或文档
- 未明确要求创建新文件时，在原文件上修改维护
