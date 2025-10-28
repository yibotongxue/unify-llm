# 统一大语言模型框架 (Unify-LLM)

[English README](./README_en.md) | [中文 README](./README.md)

这是一个统一的大语言模型推理框架，支持多种推理后端（API、Hugging Face、vLLM等），提供统一的接口和缓存管理功能。

## 项目概述

统一大语言模型框架 (Unify-LLM) 旨在为不同的大语言模型提供统一的推理接口，支持以下推理后端：

- **API 推理**: 支持多种主流 LLM API 提供商（OpenAI, Anthropic, Google, 阿里云等）
- **Hugging Face 推理**: 支持本地部署的 Hugging Face 模型
- **vLLM 推理**: 支持高性能的 vLLM 推理引擎

## 功能特性

- **统一接口**: 为不同提供商的 LLM 提供统一的调用接口
- **缓存管理**: 内置缓存机制，支持结果缓存以提高效率
- **多语言支持**: 支持中英文等多种语言模型
- **配置驱动**: 通过配置文件灵活管理模型和推理参数
- **批量推理**: 支持批量处理多个推理请求

## 安装

本项目使用 `uv` 管理依赖，推荐使用以下命令安装：

```bash
# 安装 uv（如果尚未安装）
pip3 install uv

# 安装项目依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

## 使用

### 命令行使用

```bash
uv run -m unify_llm.inference --config-file-path /path/to/config.yaml
```

### 配置文件示例

创建一个配置文件（如 `config.yaml`）来定义模型和推理参数：

```yaml
model_cfgs:
  inference_backend: "api"  # 可选: api, hf, vllm
  model_name: "gpt-4o"
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"

inference_cfgs:
  temperature: 0.7
  max_tokens: 1024

cache_cfgs:  # 可选配置
  cache_type: "redis"
  redis_host: "localhost"
  redis_port: 6379
```

### Python 代码使用

```python
from unify_llm.inference.factory import InferenceFactory
from unify_llm.utils.type_utils import InferenceInput

# 创建推理实例
model_cfgs = {
    "inference_backend": "api",
    "model_name": "gpt-4o",
    "api_key": "your-api-key"
}
inference_cfgs = {
    "temperature": 0.7,
    "max_tokens": 1024
}
cache_cfgs = None  # 不启用缓存

inference = InferenceFactory.get_inference_instance(
    model_cfgs=model_cfgs,
    inference_cfgs=inference_cfgs,
    cache_cfgs=cache_cfgs
)

# 创建输入
inference_input = [
    InferenceInput.from_prompts(
        prompt="中国的首都是哪里？",
        system_prompt="你是一个人工智能助手",
    )
]

# 执行推理
outputs = inference._generate(
    inference_input,
    enable_tqdm=True,
)
```

## 项目结构

```
unify-llm/
├── configs/           # 配置文件目录
├── outputs/           # 输出文件目录
├── logs/              # 日志文件目录
├── src/               # 源代码根目录
│   └── unify_llm/     # 源代码目录
│       ├── cache_manager/ # 缓存管理模块
│       ├── inference/     # 推理模块
│       │   ├── api_llm/   # API 模型实现
│       │   ├── hf.py      # Hugging Face 推理实现
│       │   ├── vllm.py    # vLLM 推理实现
│       │   ├── factory.py # 推理工厂类
│       │   └── base.py    # 基础推理接口
│       ├── prompts/       # 提示词管理
│       └── utils/         # 工具函数
├── tests/             # 测试文件目录
├── main.py            # 主程序入口
└── pyproject.toml     # 项目配置文件
```

## 开发配置

本项目集成了 `pre-commit` 工具，用于代码规范检查。安装依赖后运行以下命令启用：

```bash
pre-commit install
```

`pre-commit` 会在每次提交代码前自动运行以下检查：

1. **基础代码检查与修复**（来自 `pre-commit-hooks`）
   - 检查符号链接有效性
   - 删除行尾空格、确保文件末尾换行
   - 验证 YAML/TOML 文件语法
   - 阻止提交大文件（默认 >500KB）
   - 检测调试语句（如 `import pdb`）和私钥泄露
   - 检查可执行文件的 shebang 声明

2. **Python 代码格式化**
   - `isort`：自动排序 import 语句
   - `black-jupyter`：格式化 Python 和 Jupyter 笔记本代码（遵循 PEP8）

3. **Python 代码优化**
   - `autoflake`：移除未使用的 imports 和变量
   - `pyupgrade`：自动升级代码到 Python 3.12+ 语法

4. **安全检查**
   - `bandit`：扫描 Python 代码的安全漏洞（使用 `.bandit.yml` 配置）

5. **静态类型检查**
   - `mypy`：严格类型检查（Python 3.12 环境，忽略测试目录）

6. **拼写检查**
   - `codespell`：修复常见英文拼写错误

## 许可证

本项目采用[MIT许可](./LICENSE)。
