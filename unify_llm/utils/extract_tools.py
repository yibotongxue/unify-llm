import re


def extract_last_code_block(text: str, language: str = "") -> str | None:
    """
    从文本中提取最后一个代码块

    参数
    ----
    text : str
        包含代码块的文本
    language : str, 默认为空字符串
        代码块的语言标识符（如"python"）

    返回
    ----
    str | None
        提取的代码块内容，如果未找到则返回None
    """

    pattern = rf"```{language}\n(.*?)\n```" if language else r"```(?:\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches and len(matches) > 0 and isinstance(matches[-1], str):
        return matches[-1].strip()
    return None
