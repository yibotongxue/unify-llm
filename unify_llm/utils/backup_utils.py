# Backup utility functions
# Added by Qwen Code
import os
import shutil


def backup_project_files(
    output_dir: str,
    config_file_path: str,
    project_root: str,
) -> None:
    """
    备份项目文件到输出目录下的src目录中

    参数
    ----
    output_dir : str
        输出目录路径
    config_file_path : str
        配置文件路径
    project_root : str
        项目根目录路径
    """
    # 创建备份目录
    backup_dir = os.path.join(output_dir, "src")
    os.makedirs(backup_dir, exist_ok=True)

    # 备份 llm_evaluator 目录下的所有 Python 文件（除去 __pycache__）
    llm_evaluator_dir = os.path.join(project_root, "llm_evaluator")
    if os.path.exists(llm_evaluator_dir):
        shutil.copytree(
            llm_evaluator_dir,
            os.path.join(backup_dir, "llm_evaluator"),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "__init__.pyc"),
        )

    # 备份运行时使用的配置文件
    config_file_name = os.path.basename(config_file_path)
    dst_config_path = os.path.join(backup_dir, config_file_name)
    shutil.copy2(config_file_path, dst_config_path)

    # 备份 pyproject.toml
    pyproject_path = os.path.join(project_root, "pyproject.toml")
    if os.path.exists(pyproject_path):
        shutil.copy2(pyproject_path, os.path.join(backup_dir, "pyproject.toml"))
