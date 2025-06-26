# rcps/__init__.py

"""
RCPS (Reflective Coherent Presentation Synthesis) Core Package.

This package contains all the core logic for the RCPS framework, including
document parsing, narrative planning, layout generation, iterative refinement,
and final presentation rendering.

By importing key classes here, we provide a simplified public API for users of this package.
"""

# --- 核心生成器 ---
from .rcps_generator import RCPSGenerator

# --- 核心数据结构 ---
from .document import Document
from .presentation import Presentation, SlidePage
from .shapes import Shape, TextElement, ImageElement

# --- 核心组件 ---
from .document import PDFParser
from RCPS_Project.rcps.lpg.lpg import LayoutPrototypeGenerator
from .renderer import PptxRenderer
from .agent import AgentFactory

# --- 异常类 ---
from .exceptions import (
    RCPSException,
    LLMError,
    ParsingError,
    GenerationError,
    LayoutError,
    ConfigError
)

# 使用 __all__ 定义当 'from rcps import *' 时应导入的公共对象
__all__ = [
    # Classes
    'RCPSGenerator',
    'Document',
    'Presentation',
    'SlidePage',
    'Shape',
    'TextElement',
    'ImageElement',
    'PDFParser',
    'LayoutPrototypeGenerator',
    'PptxRenderer',
    'AgentFactory',
    # Exceptions
    'RCPSException',
    'LLMError',
    'ParsingError',
    'GenerationError',
    'LayoutError',
    'ConfigError',
]

# 你也可以在这里设置包级别的日志记录器，但这通常不是必需的
# from .utils import get_logger
# logger = get_logger(__name__)
# logger.info("RCPS package initialized.")