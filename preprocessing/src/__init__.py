"""Reusable helpers for the preprocessing scripts.

The scripts under `preprocessing/scripts/` currently inline their own copies
of these helpers; this package is the canonical place to import them from.
"""

from .camera import fibonacci_sphere, random_sphere, look_at_c2w
from .projection import load_txt_point_cloud, project_points_fullscene

__all__ = [
    "fibonacci_sphere",
    "random_sphere",
    "look_at_c2w",
    "load_txt_point_cloud",
    "project_points_fullscene",
]
