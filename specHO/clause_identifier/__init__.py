"""
Clause Identifier Module

This module identifies clause boundaries and creates thematic pairs for echo analysis.

Components:
- boundary_detector.py: Detects clause boundaries using dependency parsing
- pair_rules.py: Applies pairing rules (punctuation, conjunction, transition)
- zone_extractor.py: Extracts terminal/initial zones from clause pairs
- pipeline.py: Orchestrates clause identification process
"""

__all__ = ["ClauseBoundaryDetector"]
