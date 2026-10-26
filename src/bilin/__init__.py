"""Bilinear idealisation of force-deformation (capacity) curves."""

from bilin.bilinearization import (
    BilinearizationError,
    BilinOptions,
    BilinResult,
    Iteration,
    bilinearize,
)
from bilin.curves import BilinearCurve, Curve

__all__ = [
    "BilinOptions",
    "BilinResult",
    "BilinearCurve",
    "BilinearizationError",
    "Curve",
    "Iteration",
    "bilinearize",
]
