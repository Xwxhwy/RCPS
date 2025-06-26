# rcps/shapes.py (修正版)
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal


@dataclass
class Shape:
    element_id: str
    shape_type: Literal["text_box", "image", "auto_shape", "chart", "table"]

    left: float = 0.0
    top: float = 0.0
    width: float = 100.0
    height: float = 100.0
    rotation: float = 0.0

    # 内容属性
    content: Optional[Any] = None

    # 样式属性
    style: Dict[str, Any] = field(default_factory=dict)

    # z_order决定了元素的堆叠顺序
    z_order: int = 0


@dataclass
class TextElement(Shape):
    """专门用于表示文本框的元素。"""
    shape_type: Literal["text_box"] = "text_box"
    content: str = ""  # 明确内容为字符串


@dataclass
class ImageElement(Shape):
    """专门用于表示图像的元素。"""
    shape_type: Literal["image"] = "image"
    content: str = ""  # 明确内容为图像路径