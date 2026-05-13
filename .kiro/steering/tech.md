# 技术栈

## 编程语言

- Python 3.8+

## 核心框架与库

- **FunASR**: 阿里达摩院开源的端到端语音识别工具包，提供 Paraformer 模型基础架构
- **PyTorch**: 深度学习框架
- **ModelScope**: 模型管理与推理平台
- **NumPy**: 数值计算
- **SoundFile / librosa**: 音频文件读取与处理
- **ONNX / ONNX Runtime**: 模型导出与高性能推理（可选）

## 构建与依赖管理

- `pip` + `requirements.txt` 管理 Python 依赖
- 建议使用虚拟环境（venv 或 conda）

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行推理
python inference.py --audio <音频路径> --hotwords <热词列表>

# 运行测试
pytest tests/

# 代码格式检查
flake8 src/
```

## 开发规范

- Python 文件名必须使用英文命名
- 文档及日志输出统一使用中文
- 功能代码完成后不自动生成测试文件或文档
- 未明确要求创建新文件时，在原文件上修改维护
