# Unified Large Language Model Framework (Unify-LLM)

[English README](./README_en.md) | [中文 README](./README.md)

This is a unified large language model inference framework that supports multiple inference backends (API, Hugging Face, vLLM, etc.) and provides unified interfaces with cache management functionality.

## Project Overview

The Unified Large Language Model Framework (Unify-LLM) aims to provide unified inference interfaces for different large language models, supporting the following inference backends:

- **API Inference**: Supports various mainstream LLM API providers (OpenAI, Anthropic, Google, Alibaba Cloud, etc.)
- **Hugging Face Inference**: Supports locally deployed Hugging Face models
- **vLLM Inference**: Supports high-performance vLLM inference engine

## Features

- **Unified Interface**: Provides a unified calling interface for different LLM providers
- **Cache Management**: Built-in caching mechanism that supports result caching to improve efficiency
- **Multi-language Support**: Supports Chinese, English, and other language models
- **Configuration-driven**: Flexible management of models and inference parameters via configuration files
- **Batch Inference**: Supports batch processing of multiple inference requests

## Installation

This project uses `uv` to manage dependencies. It is recommended to install using the following commands:

```bash
# Install uv (if not already installed)
pip3 install uv

# Install project dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Usage

### Command Line Usage

```bash
python -m unify_llm.inference --config-file-path /path/to/config.yaml
```

### Configuration File Example

Create a configuration file (e.g., `config.yaml`) to define models and inference parameters:

```yaml
model_cfgs:
  inference_backend: "api"  # Options: api, hf, vllm
  model_name: "gpt-4o"
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"

inference_cfgs:
  temperature: 0.7
  max_tokens: 1024

cache_cfgs:  # Optional configuration
  cache_type: "redis"
  redis_host: "localhost"
  redis_port: 6379
```

### Python Code Usage

```python
from unify_llm.inference.factory import InferenceFactory
from unify_llm.utils.type_utils import InferenceInput

# Create inference instance
model_cfgs = {
    "inference_backend": "api",
    "model_name": "gpt-4o",
    "api_key": "your-api-key"
}
inference_cfgs = {
    "temperature": 0.7,
    "max_tokens": 1024
}
cache_cfgs = None  # Disable cache

inference = InferenceFactory.get_inference_instance(
    model_cfgs=model_cfgs,
    inference_cfgs=inference_cfgs,
    cache_cfgs=cache_cfgs
)

# Create input
inference_input = [
    InferenceInput.from_prompts(
        prompt="Where is the capital of China?",
        system_prompt="You are an AI assistant",
    )
]

# Perform inference
outputs = inference._generate(
    inference_input,
    enable_tqdm=True,
)
```

## Project Structure

```
unify-llm/
├── configs/           # Configuration files directory
├── outputs/           # Output files directory
├── logs/              # Log files directory
├── unify_llm/         # Source code directory
│   ├── cache_manager/ # Cache management module
│   ├── inference/     # Inference module
│   │   ├── api_llm/   # API model implementation
│   │   ├── hf.py      # Hugging Face inference implementation
│   │   ├── vllm.py    # vLLM inference implementation
│   │   ├── factory.py # Inference factory class
│   │   └── base.py    # Base inference interface
│   ├── prompts/       # Prompt management
│   └── utils/         # Utility functions
├── tests/             # Test files directory
├── main.py            # Main program entry
└── pyproject.toml     # Project configuration file
```

## Development Configuration

This project integrates the `pre-commit` tool for code style checking. After installing dependencies, run the following command to enable:

```bash
pre-commit install
```

`pre-commit` will automatically run the following checks before each code commit:

1. **Basic code checking and fixing** (from `pre-commit-hooks`)
   - Check symbolic link validity
   - Remove trailing whitespaces and ensure file ends with newline
   - Validate YAML/TOML file syntax
   - Prevent large file commits (default >500KB)
   - Detect debug statements (e.g. `import pdb`) and private key leaks
   - Check executable file shebang declarations

2. **Python code formatting**
   - `isort`: Automatically sort import statements
   - `black-jupyter`: Format Python and Jupyter notebook code (following PEP8)

3. **Python code optimization**
   - `autoflake`: Remove unused imports and variables
   - `pyupgrade`: Automatically upgrade code to Python 3.12+ syntax

4. **Security checking**
   - `bandit`: Scan Python code for security vulnerabilities (using `.bandit.yml` configuration)

5. **Static type checking**
   - `mypy`: Strict type checking (Python 3.12 environment, ignoring test directory)

6. **Spell checking**
   - `codespell`: Fix common English spelling errors

## License

This project is licensed under the [MIT License](./LICENSE).
