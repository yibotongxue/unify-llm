# Python项目模板

这是一个Python项目的模板，可以帮助建立一个初始的规范开发环境。

## 使用

点击左上方 `use this template` 按钮，使用本模板建立一个自己的仓库，拉取到本地，即可继续开发。本项目使用 `uv` 管理依赖，项目本身也是使用命令

```bash
uv init python-project-template --python=3.12
```

命令创建的。

使用命令

```bash
uv sync
```

安装开发依赖，如果没有安装 `uv` 的，可以使用命令

```bash
pip3 install uv
```

安装。 `uv` 工具的相关内容可以参考[文档](https://docs.astral.sh/uv/)或[中文文档](https://uv.doczh.com/)。

激活环境

```bash
source .venv/bin/activate
```

安装 `pre-commit` 工具

```bash
pre-commit install
```

`pre-commit` 工具的相关内容可以参考[文档](https://pre-commit.com/)。使用这个工具，会在每次git commit之前进行一些检查。我们的检查配置文件在[.pre-commit-config.yaml](./.pre-commit-config.yaml)，相关地说明参考[配置文件说明](#pre-commit-配置文件说明)。

## 适配

一般的，可以修改 `src` 为自己的目录名，并同步修改[pyproject.toml](./pyproject.toml)中的相关内容。可以将[pyproject.toml](./pyproject.toml)中的如下内容

```toml
[project]
name = "python-project-template"
version = "0.1.0"
description = "Python项目模板"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []
```

中的项目名为自己的项目名，对应地修改项目描述，按照需要调整版本号等。

## `pre-commit` 配置文件说明

这里直接复制[DeepSeek](https://deepseek.com/)的说明：

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
