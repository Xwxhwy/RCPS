import fitz
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .utils import get_logger, ensure_dir
from .exceptions import ParsingError

logger = get_logger(__name__)


@dataclass
class BoundingBox:
    """标准化的边界框对象，单位为磅(points)。"""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

# --- 修复开始：重新设计基类和子类 ---

@dataclass
class DocumentElement:
    element_id: str
    page_num: int
    bbox: BoundingBox

@dataclass
class TextBlock(DocumentElement):
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MediaObject(DocumentElement):
    path: str
    caption_llm: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- 修复结束 ---


# ==============================================================================
# 2. 文档容器 (Document Container)
# ==============================================================================

@dataclass
class Document:
    """一个完整的、已解析文档的内存表示。"""
    doc_id: str
    source_path: str
    title: Optional[str]
    elements: List[DocumentElement] = field(default_factory=list)
    page_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_full_text(self, separator: str = "\n") -> str:
        """获取并拼接文档中的所有文本内容。"""
        text_parts = [elem.text for elem in self.elements if isinstance(elem, TextBlock)]
        return separator.join(text_parts)

    def get_elements_by_page(self, page_num: int) -> List[DocumentElement]:
        """获取指定页面的所有元素。"""
        return [elem for elem in self.elements if elem.page_num == page_num]

    def get_element_by_id(self, element_id: str) -> Optional[DocumentElement]:
        """通过ID获取一个元素。"""
        for elem in self.elements:
            if elem.element_id == element_id:
                return elem
        return None


class PDFParser:
    """一个高效的PDF解析器。"""

    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        self.images_dir = os.path.join(self.workspace_dir, "images")
        ensure_dir(self.images_dir)
        logger.info(f"PDFParser initialized. Image output will be saved to: {self.images_dir}")

    def parse(self, file_path: str) -> Document:
        """解析一个PDF文件。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        elements: List[DocumentElement] = []

        try:
            pdf_doc = fitz.open(file_path)
        except Exception as e:
            raise ParsingError(f"Failed to open or parse PDF file: {file_path}.") from e

        for page_num_zero_based, page in enumerate(pdf_doc):
            page_num = page_num_zero_based + 1

            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        for span in line['spans']:
                            text = span['text'].strip()
                            if text:
                                bbox = BoundingBox(x0=span['bbox'][0], y0=span['bbox'][1], x1=span['bbox'][2],
                                                   y1=span['bbox'][3])
                                element_id = f"{doc_id}_p{page_num}_b{block['number']}_l{line['wmode']}_s{span['size']}"
                                elements.append(
                                    TextBlock(element_id=element_id, page_num=page_num, bbox=bbox, text=text))

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base_image = pdf_doc.extract_image(xref)
                    if not base_image: continue

                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    media_id = f"{doc_id}_p{page_num}_img{img_index}"
                    image_filename = f"{media_id}.{image_ext}"
                    image_path = os.path.join(self.images_dir, image_filename)

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    img_bbox_tuple = page.get_image_bbox(img)
                    bbox = BoundingBox(x0=img_bbox_tuple[0], y0=img_bbox_tuple[1], x1=img_bbox_tuple[2],
                                       y1=img_bbox_tuple[3])
                    elements.append(MediaObject(element_id=media_id, page_num=page_num, bbox=bbox, path=image_path))
                except Exception as e:
                    logger.warning(
                        f"Could not extract image with xref {xref} on page {page_num}. Skipping. Reason: {e}")

        doc = Document(
            doc_id=doc_id,
            source_path=file_path,
            title=pdf_doc.metadata.get("title", doc_id),
            elements=elements,
            page_count=pdf_doc.page_count,
            metadata=pdf_doc.metadata
        )
        pdf_doc.close()

        logger.info(
            f"Successfully parsed '{file_path}': Found {len([e for e in elements if isinstance(e, TextBlock)])} text blocks and {len([e for e in elements if isinstance(e, MediaObject)])} media objects.")

        return doc