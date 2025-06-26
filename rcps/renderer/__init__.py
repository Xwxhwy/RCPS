# rcps/renderer/__init__.py
"""
Renderer Sub-package.

This package is responsible for converting the in-memory Presentation object
into a final, viewable file format, such as .pptx.
"""
from .base_renderer import BaseRenderer
from .pptx_renderer import PptxRenderer
from .layout_interpreter import LayoutInterpreter

__all__ = ['BaseRenderer', 'PptxRenderer', 'LayoutInterpreter']