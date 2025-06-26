# rcps/renderer/layout_interpreter.py (已修正)

from typing import List, Tuple, Dict, Optional, Any  # <-- 已修正：在这里添加了 Any
from .. import constants as C
from ..shapes import Shape, TextElement, ImageElement  # 假设这些已定义
from ..utils import get_logger  # 假设你有这个工具

logger = get_logger(__name__)


class LayoutRegion:
    """定义一个可用于布局计算的矩形区域。"""

    def __init__(self, left, top, width, height):
        self.left, self.top, self.width, self.height = left, top, width, height

    def get_bbox(self) -> Tuple[float, float, float, float]:
        return self.left, self.top, self.width, self.height

    def split_vertical(self, ratio: float = 0.5, gap: float = 0) -> Tuple['LayoutRegion', 'LayoutRegion']:
        """垂直切分区域。"""
        left_width = self.width * ratio - gap / 2
        right_width = self.width * (1 - ratio) - gap / 2
        left_region = LayoutRegion(self.left, self.top, left_width, self.height)
        right_region = LayoutRegion(self.left + left_width + gap, self.top, right_width, self.height)
        return left_region, right_region

    def split_horizontal(self, ratio: float = 0.5, gap: float = 0) -> Tuple['LayoutRegion', 'LayoutRegion']:
        """水平切分区域。"""
        top_height = self.height * ratio - gap / 2
        bottom_height = self.height * (1 - ratio) - gap / 2
        top_region = LayoutRegion(self.left, self.top, self.width, top_height)
        bottom_region = LayoutRegion(self.left, self.top + top_height + gap, self.width, bottom_height)
        return top_region, bottom_region


class LayoutInterpreter:
    """
    采用更先进的、基于“区域管理”和“动态约束”的策略，将LDL序列精确地转化为具体的几何坐标。它能更好地处理复杂的、嵌套的布局，并充分利用LDL中丰富的属性和位置token。
    """

    def __init__(self, page_width: float = 720, page_height: float = 405):
        self.PAGE_WIDTH = page_width
        self.PAGE_HEIGHT = page_height
        self.PADDING = page_width * 0.05

    def interpret(self, page_id: int, ldl_tokens: List[str], slide_content: Dict) -> List[Shape]:
        layout_type, element_blocks = self._parse_ldl(ldl_tokens)
        regions = self._initialize_layout_regions(layout_type, len(element_blocks))
        shapes: List[Shape] = []
        z_order_counter = 0
        for i, block in enumerate(element_blocks):
            element_id = f"p{page_id}_elem{i}"

            # 确定元素类型和内容
            shape_type, content = self._get_content_for_block(block, slide_content)
            if not shape_type: continue

            base_region = regions.get(i, regions['default'])

            # 根据块内的POS_*和ATTR_* token，在基础区域内进行微调和约束计算
            bbox = self._calculate_bbox_with_constraints(block, base_region)

            # 创建Shape对象
            shape_class = TextElement if shape_type == "text" else ImageElement
            shapes.append(shape_class(
                element_id=element_id, content=content,
                left=bbox[0], top=bbox[1], width=bbox[2], height=bbox[3],
                z_order=z_order_counter
            ))
            z_order_counter += 1

        return shapes

    def _parse_ldl(self, tokens: List[str]) -> Tuple[str, List[List[str]]]:
        """解析原始token列表，分离出主布局类型和元素块。"""
        # 提取主布局类型
        layout_type = next((t for t in tokens if t.startswith("LAYOUT_")), C.LAYOUT_CONTENT_ONE_COL)

        # 移除布局和特殊token，准备切分元素块
        elem_tokens = [t for t in tokens if t not in [C.SOS_TOKEN, C.EOS_TOKEN] and not t.startswith("LAYOUT_")]

        blocks, current_block = [], []
        for token in elem_tokens:
            if token == C.SEP_TOKEN:
                if current_block: blocks.append(current_block)
                current_block = []
            else:
                current_block.append(token)
        if current_block: blocks.append(current_block)

        return layout_type, blocks

    def _initialize_layout_regions(self, layout_type: str, num_blocks: int) -> Dict[Any, LayoutRegion]:
        W, H, P = self.PAGE_WIDTH, self.PAGE_HEIGHT, self.PADDING
        full_page = LayoutRegion(P, P, W - 2 * P, H - 2 * P)
        regions = {'default': full_page}

        title_region, main_content_region = full_page.split_horizontal(ratio=0.2, gap=P * 0.5)

        if layout_type == C.LAYOUT_TITLE_SUBTITLE:
            regions[0] = title_region
            regions[1] = main_content_region

        elif layout_type == C.LAYOUT_CONTENT_TWO_COL:
            # 假设第0个块是标题，剩下的是两栏内容
            regions[0] = title_region
            left_col, right_col = main_content_region.split_vertical(gap=P)
            # 将剩余的块均匀分配到两栏
            content_blocks = num_blocks - 1
            for i in range(1, num_blocks):
                regions[i] = left_col if (i - 1) < content_blocks / 2 else right_col

        elif layout_type == C.LAYOUT_IMAGE_WITH_CAPTION:
            # 假设第0个是图片，第1个是说明文字
            image_region, caption_region = main_content_region.split_horizontal(ratio=0.8, gap=P * 0.2)
            regions[0] = image_region
            regions[1] = caption_region


        return regions

    def _calculate_bbox_with_constraints(self, block: List[str], region: LayoutRegion) -> Tuple[
        float, float, float, float]:

        l, t, w, h = region.get_bbox()

        # 默认使用整个区域
        final_bbox = [l, t, w, h]

        # 解析尺寸属性 (ATTR_SIZE_*)
        if C.ATTR_SIZE_MAJOR in block:
            pass  # 默认占满区域
        elif C.ATTR_SIZE_MINOR in block:
            # 次要元素尺寸缩小
            final_bbox[2] *= 0.8
            final_bbox[3] *= 0.8

        # 解析位置属性 (POS_*) - 这部分可以实现得非常精细
        if C.POS_TOP in block:
            final_bbox[1] = t
        if C.POS_BOTTOM in block:
            final_bbox[1] = t + h - final_bbox[3]
        if C.POS_LEFT in block:
            final_bbox[0] = l
        if C.POS_RIGHT in block:
            final_bbox[0] = l + w - final_bbox[2]
        if C.POS_CENTER in block:  # 水平居中
            final_bbox[0] = l + (w - final_bbox[2]) / 2
        if C.POS_MIDDLE in block:  # 垂直居中
            final_bbox[1] = t + (h - final_bbox[3]) / 2

        if C.ELEM_IMAGE in block:
            # 假设可以获取图片原始宽高比
            aspect_ratio = 16 / 9
            if C.ATTR_IMAGE_ASPECT_TALL in block: aspect_ratio = 9 / 16
            if C.ATTR_IMAGE_ASPECT_SQUARE in block: aspect_ratio = 1 / 1

            bbox_w, bbox_h = final_bbox[2], final_bbox[3]
            if bbox_w / bbox_h > aspect_ratio:  # 框太宽
                new_w = bbox_h * aspect_ratio
                final_bbox[0] += (bbox_w - new_w) / 2  # 居中
                final_bbox[2] = new_w
            else:  # 框太高
                new_h = bbox_w / aspect_ratio
                final_bbox[1] += (bbox_h - new_h) / 2  # 居中
                final_bbox[3] = new_h

        return tuple(final_bbox)

    def _get_content_for_block(self, block: List[str], slide_content: Dict) -> Tuple[Optional[str], Any]:
        content_map = {
            C.ELEM_TITLE: ("text", slide_content.get("title")),
            C.ELEM_SUBTITLE: ("text", slide_content.get("subtitle")),
            C.ELEM_BODY_TEXT: ("text", "\n".join(slide_content.get("body", []))),
            C.ELEM_BULLET_POINTS: ("text", "\n".join(slide_content.get("bullet_points", []))),
            C.ELEM_IMAGE: ("image", slide_content.get("image_path")),
            C.ELEM_CHART: ("image", slide_content.get("chart_path")),  # 图表也视为图片
        }

        elem_type_token = block[0]
        return content_map.get(elem_type_token, (None, None))