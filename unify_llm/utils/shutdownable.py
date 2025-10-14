class Shutdownable:
    def __init__(self) -> None:
        self._is_shutting_down = False  # 标记是否正在关闭中

    def shutdown(self) -> None:
        """关闭当前对象及其所有 Shutdownable 属性"""
        if getattr(self, "_is_shutting_down", False):
            return  # 如果已在关闭过程中，直接返回避免递归
        self._is_shutting_down = True  # 标记开始关闭

        try:
            # 遍历所有属性
            for attr_name in dir(self):
                # 跳过特殊方法（如 __init__, __str__ 等）
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue

                try:
                    # 获取属性值
                    attr_value = getattr(self, attr_name)
                except AttributeError:
                    continue  # 忽略获取属性时的异常

                # 检查属性是否为 Shutdownable 实例且不是自身
                if isinstance(attr_value, Shutdownable) and attr_value is not self:
                    attr_value.shutdown()  # 递归关闭
        finally:
            self._is_shutting_down = False  # 重置关闭标记
