# 项目目录结构

```
SeACoParaformer/
├── src/                # 代码目录 - 核心源代码
│   ├── model/          # 模型定义（Paraformer、SeACo 模块等）
│   ├── data/           # 数据处理与加载
│   ├── utils/          # 工具函数
│   └── inference/      # 推理相关代码
├── docs/               # 文档目录 - 项目文档（中文）
├── tests/              # 测试目录 - 单元测试与集成测试
├── configs/            # 配置文件（模型参数、训练配置等）
├── scripts/            # 脚本（训练、评估、数据准备等）
├── models/             # 预训练模型存放（不纳入版本控制）
├── data/               # 数据集存放（不纳入版本控制）
├── requirements.txt    # Python 依赖
├── README.md           # 项目说明（中文）
└── .kiro/              # Kiro 配置
    └── steering/       # 引导规则
```

## 目录职责

| 目录 | 用途 |
|------|------|
| `src/` | 所有功能代码，按模块划分子目录 |
| `docs/` | 设计文档、使用说明、API 文档等 |
| `tests/` | 测试代码，与 src 结构对应 |
| `configs/` | YAML/JSON 配置文件 |
| `scripts/` | 独立可执行脚本（训练、评估、部署） |

## 命名规则

- 目录名：英文小写，下划线分隔
- Python 文件名：英文小写，下划线分隔（如 `seaco_decoder.py`）
- 文档文件：可使用中文文件名
