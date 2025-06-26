# rcps/preval/__init__.py
"""
PREVAL (Preference-based Evaluation) Sub-package.

This package provides tools to evaluate the quality of a generated presentation
based on a pre-trained, multi-dimensional preference model.
"""
from .evaluator import PREVAL
from .feature_extractor import PrevalFeatureExtractor

__all__ = ['PREVAL', 'PrevalFeatureExtractor']