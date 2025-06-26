# rcps/lpg/__init__.py

"""
LPG (Layout Prototype Generation) Sub-package.

This package encapsulates all logic related to generating symbolic layout
descriptions (LDL) from abstract slide concepts.
"""
from .generator import LayoutPrototypeGenerator
from .feature_extractor import LPGFeatureExtractor

__all__ = ['LayoutPrototypeGenerator', 'LPGFeatureExtractor']