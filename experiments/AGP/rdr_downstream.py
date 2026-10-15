"""Compatibility imports for the frozen four-generator community source.

Only its biological feature dependencies are included in this publication.
"""
try:
    from .four_generator_features import normalize, build_groups, feature_matrix, rules
except ImportError:
    from four_generator_features import normalize, build_groups, feature_matrix, rules

__all__ = ["normalize", "build_groups", "feature_matrix", "rules"]
