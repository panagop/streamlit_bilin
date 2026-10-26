"""Iterative equal-energy bilinearisation of force-deformation curves.

Ported unchanged (numerically) from ``streng.tools.bilin`` (streng 0.0.7), which no
longer imports on Python >= 3.11. Procedure:

1. Truncate the curve at ``min(x at strength drop, xtarget)``; ``xtarget = 0``
   keeps the whole curve.
2. Shift the curve so that it starts at x = 0.
3. Start from the secant stiffness at 0.2·Vmax.
4. Iterate: y_u = (2·Vmax + V_end)/3, x_u = x_end; x_y from equal areas;
   update k_el as the secant stiffness at 0.6·V_y, until convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bilin.curves import BilinearCurve, Curve, FloatArray

MAX_ITERATIONS = 1000


class BilinearizationError(ValueError):
    """Raised when the procedure gives a non-physical idealisation."""


@dataclass(frozen=True)
class BilinOptions:
    xtarget: float = 0.0
    """Deformation up to which the curve is used (0 = whole curve)."""
    dropstrength: float = 0.75
    """Curve is cut at the first point after Vmax with V <= dropstrength·Vmax."""
    elastoplastic: bool = False
    """Force a zero post-yield stiffness."""
    allowa010: bool = True
    """Allow hardening ratio α = k_inel/k_el > 0.10; if False, α is capped at 0.10."""
    tolerance: float = 1e-6
    """Convergence tolerance on the relative change of k_el."""


@dataclass(frozen=True)
class Iteration:
    number: int
    x_y: float
    y_y: float
    x_u: float
    y_u: float
    kinel: float
    kel: float
    k_06: float
    error: float


@dataclass(frozen=True)
class BilinResult:
    curve: BilinearCurve
    x_used: FloatArray
    """Truncated curve used in the calculation (not shifted)."""
    y_used: FloatArray
    area: float
    k_02: float
    iterations: list[Iteration] = field(default_factory=list)


def _xy_with_endpoints(
    x: FloatArray, y: FloatArray, x_start: float, x_end: float
) -> tuple[FloatArray, FloatArray]:
    """Clip the curve to [x_start, x_end], interpolating the end points."""
    x_start = max(x_start, float(x[0]))
    x_end = min(x_end, float(x[-1]))

    y_start = float(np.interp(x_start, x, y))
    xs = x[x > x_start]
    ys = y[len(y) - len(xs) :]
    xs = np.append(x_start, xs)
    ys = np.append(y_start, ys)

    y_end = float(np.interp(x_end, x, y))
    xs = xs[xs < x_end]
    ys = ys[: len(xs)]
    return np.append(xs, x_end), np.append(ys, y_end)


def _truncation_point(curve: Curve, options: BilinOptions) -> float:
    ymax = float(np.max(curve.y))
    i_max = int(np.argmax(curve.y))
    y_tail = curve.y[i_max:]
    x_tail = curve.x[i_max:]
    i = 0
    while y_tail[i] > options.dropstrength * ymax and i < len(y_tail) - 1:
        i += 1
    return min(float(x_tail[i]), options.xtarget)


def _iteration(
    x: FloatArray, y: FloatArray, kel: np.float64, area: np.float64, options: BilinOptions
) -> tuple[float, float, float, float, float, float]:
    # numpy scalars on purpose: division by zero or sqrt of a negative number gives
    # inf/NaN (as in streng) instead of raising, and the result is checked afterwards.
    last = len(x) - 1
    ymax = np.max(y)

    y_u = (2 * ymax + y[last]) / 3.0
    x_u = x[last]
    x_y = (2 * area - x_u * y_u) / (kel * x_u - y_u)
    y_y = kel * x_y
    kinel = (y_u - y_y) / (x_u - x_y)

    # α capped at 0.10
    if not options.allowa010 and kinel / kel > 0.1:
        b = -2.0 * x_u * kel
        c = 1.8 * area * kel + 0.1 * (kel * x_u) ** 2
        y_u = (-b - np.sqrt(b * b - 4 * c)) / 2.0
        x_y = (y_u - 0.1 * kel * x_u) / (0.9 * kel)
        y_y = kel * x_y

    # Elastic-perfectly plastic, or negative post-yield stiffness
    if options.elastoplastic or kinel < 0.0:
        b = -2.0 * x_u
        c = 2.0 * area / kel
        x_y = (-b - np.sqrt(b * b - 4 * c)) / 2.0
        y_y = x_y * kel
        y_u = y_y

    y_06 = 0.6 * y_y
    x_06 = np.interp(y_06, y, x)
    k_06 = y_06 / x_06
    return float(x_y), float(y_y), float(x_u), float(y_u), float(kinel), float(k_06)


def bilinearize(curve: Curve, options: BilinOptions | None = None) -> BilinResult:
    """Bilinear idealisation of ``curve``.

    Raises:
        BilinearizationError: if the result is non-physical (e.g. d_y < d_0 or NaN),
            which happens for curves with an abrupt strength drop at the end.
    """
    opts = options or BilinOptions()
    x0 = float(curve.x[0])

    x_target = _truncation_point(curve, opts)
    if x_target == 0.0:
        x_used, y_used = curve.x, curve.y
    else:
        x_used, y_used = _xy_with_endpoints(curve.x, curve.y, x0, x_target)
    x = x_used - x0
    y = y_used

    y02 = 0.2 * float(np.max(y))
    k02 = y02 / float(np.interp(y02, y, x))
    area = float(np.trapezoid(y, x))

    iterations: list[Iteration] = []
    kel = k02
    error = 100.0
    result = BilinearCurve(dy=0.0, ay=0.0, du=0.0, au=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        while error > opts.tolerance:
            x_y, y_y, x_u, y_u, kinel, k_06 = _iteration(x, y, np.float64(kel), np.float64(area), opts)
            error = float(np.abs((np.float64(kel) - k_06) / k_06))
            iterations.append(Iteration(len(iterations) + 1, x_y, y_y, x_u, y_u, kinel, kel, k_06, error))
            result = BilinearCurve(dy=x_y + x0, ay=y_y, du=x_u + x0, au=y_u, d0=x0, a0=0.0)
            kel = k_06
            if len(iterations) > MAX_ITERATIONS:
                break

    if not result.is_admissible:
        raise BilinearizationError(
            "Non-physical bilinear curve "
            f"(d_y={result.dy:.4g}, V_y={result.ay:.4g}); "
            "check for an abrupt strength drop near the end of the curve or adjust xtarget."
        )
    return BilinResult(result, x_used, y_used, area, k02, iterations)
