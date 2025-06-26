from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .shapes import Shape
from .utils import get_logger
from . import constants as C
from .exceptions import LayoutError

logger = get_logger(__name__)


@dataclass
class SlidePage:
    """代表一个独立的幻灯片页面及其所有待渲染的Shape元素。"""
    page_id: int
    title: str  # 每页幻灯片都有一个来自规划阶段的标题
    shapes: List[Shape] = field(default_factory=list)
    notes: Optional[str] = None  # 演讲者备注

    @classmethod
    def from_plan_and_layout(cls, page_id: int, slide_plan: Dict, layout_tokens: List[str]) -> "SlidePage":
        """
        [核心创新] 布局解释器：根据R-CoT的幻灯片计划和LPG生成的LDL，创建SlidePage实例。
        这是将符号化布局转化为具体对象的地方。
        """
        logger.info(f"Interpreting layout for slide {page_id}: {' '.join(layout_tokens)}")

        page = cls(page_id=page_id, title=slide_plan.get("title", f"Slide {page_id}"))


        try:
            # 移除 <SOS> 和 <EOS>
            tokens = [t for t in layout_tokens if t not in [C.SOS_TOKEN, C.EOS_TOKEN]]

            # 按 <SEP> 分割成元素描述块
            element_blocks = []
            current_block = []
            for token in tokens:
                if token == C.SEP_TOKEN:
                    if current_block:
                        element_blocks.append(current_block)
                    current_block = []
                else:
                    current_block.append(token)
            if current_block:
                element_blocks.append(current_block)

            # 模拟的页面尺寸 (单位: 磅)
            PAGE_WIDTH, PAGE_HEIGHT = 720, 405  # 16:9 in points (10x5.625 inches)

            # 为每个元素块创建Shape
            z_order_counter = 0
            for i, block in enumerate(element_blocks):
                element_id = f"p{page_id}_elem{i}"

                # 第一个token通常是元素类型
                elem_type_token = block[0]
                shape_type = ""
                content = None

                if elem_type_token == C.ELEM_TITLE:
                    shape_type = "text_box"
                    content = slide_plan.get("title", "Default Title")
                elif elem_type_token == C.ELEM_BODY_TEXT or elem_type_token == C.ELEM_BULLET_POINTS:
                    shape_type = "text_box"
                    content = "\n".join(slide_plan.get("bullet_points", ["Default content."]))
                elif elem_type_token == C.ELEM_IMAGE:
                    shape_type = "image"
                    content = slide_plan.get("image_path")  # 这是图像路径
                else:
                    logger.warning(f"Unknown element token '{elem_type_token}' in layout. Skipping.")
                    continue

                if shape_type == "image" and not content:
                    logger.warning(
                        f"Layout specified an image, but no image_path found in slide_plan. Skipping image element.")
                    continue

                # 解析位置和大小属性 (这是一个简化的规则引擎)
                left, top, width, height = cls._interpret_geometry(block, PAGE_WIDTH, PAGE_HEIGHT)

                page.shapes.append(Shape(
                    element_id=element_id,
                    shape_type=shape_type,
                    content=content,
                    left=left, top=top, width=width, height=height,
                    z_order=z_order_counter
                ))
                z_order_counter += 1

        except Exception as e:
            logger.error(f"Failed to interpret layout for slide {page_id}. LDL: {' '.join(layout_tokens)}. Error: {e}")
            raise LayoutError("Layout interpretation failed.") from e

        return page

    @staticmethod
    def _interpret_geometry(tokens: List[str], page_w: float, page_h: float) -> Tuple[float, float, float, float]:
        # 默认值
        left, top, width, height = page_w * 0.1, page_h * 0.1, page_w * 0.8, page_h * 0.8  # 默认居中80%

        if C.POS_TOP in tokens:
            top = page_h * 0.05
            height = page_h * 0.2
        if C.POS_MIDDLE in tokens:
            top = page_h * 0.25
            height = page_h * 0.6
        if C.POS_LEFT in tokens:
            left = page_w * 0.05
            width = page_w * 0.4
        if C.POS_RIGHT in tokens:
            left = page_w * 0.55
            width = page_w * 0.4

        return left, top, width, height


@dataclass
class Presentation:
    presentation_id: str
    slides: List[SlidePage] = field(default_factory=list)
    design_style: str = C.DESIGN_STYLE_MODERN  # 默认设计风格
    color_palette: Dict[str, str] = C.COLOR_PALETTE_CORPORATE_BLUE  # 默认色板

    def add_slide(self, slide: SlidePage):
        self.slides.append(slide)

    def get_slide_by_id(self, page_id: int) -> Optional[SlidePage]:
        for slide in self.slides:
            if slide.page_id == page_id:
                return slide
        return None