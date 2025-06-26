# rcps/renderer/base_renderer.py
from abc import ABC, abstractmethod
from ..presentation import Presentation

class BaseRenderer(ABC):
    """渲染器的抽象基类，定义了统一的渲染接口。"""
    @abstractmethod
    def render(self, presentation: Presentation, output_path: str):
        """将内存中的Presentation对象渲染到指定路径的文件。"""
        pass