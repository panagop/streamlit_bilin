"""Force-deformation curve input and the bilinear curve result."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Curve:
    """A force-deformation (capacity) curve given as x (deformation) and y (force)."""

    x: FloatArray
    y: FloatArray

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be 1-D arrays of equal length")
        if len(self.x) < 3:
            raise ValueError("a curve needs at least 3 points")

    @classmethod
    def from_text(cls, text: str) -> Curve:
        """Parse two numeric columns separated by spaces, tabs, ';' or ','.

        Extra columns are ignored. Lines starting with '#' are comments.
        """
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        delimiter = ";" if ";" in lines[0] else ("," if "," in lines[0] else None)
        data = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=delimiter, usecols=(0, 1))
        return cls(x=data[:, 0].astype(np.float64), y=data[:, 1].astype(np.float64))

    @classmethod
    def from_file(cls, path: str | Path) -> Curve:
        return cls.from_text(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class BilinearCurve:
    """Bilinear idealisation (d0, a0) -> (dy, ay) -> (du, au)."""

    dy: float
    ay: float
    du: float
    au: float
    d0: float = 0.0
    a0: float = 0.0

    @property
    def mu(self) -> float:
        """Ductility μ = (du - d0) / (dy - d0)."""
        return (self.du - self.d0) / (self.dy - self.d0)

    @property
    def kel(self) -> float:
        """Elastic stiffness k_el = (ay - a0) / (dy - d0)."""
        return (self.ay - self.a0) / (self.dy - self.d0)

    @property
    def kinel(self) -> float:
        """Post-yield stiffness k_inel = (au - ay) / (du - dy)."""
        return (self.au - self.ay) / (self.du - self.dy)

    @property
    def alpha(self) -> float:
        """Hardening ratio α = k_inel / k_el."""
        return self.kinel / self.kel

    @property
    def beta0(self) -> float:
        """Hysteretic damping, FEMA 440 Eq. 3-6."""
        mu, a = self.mu, self.alpha
        return (2 / math.pi) * ((mu - 1) * (1 - a)) / (mu * (1 + a * mu - a))

    @property
    def T0(self) -> float:
        """Initial period 2π·sqrt((dy - d0)/(ay - a0)); meaningful only for (Sd, Sa) curves."""
        return 2 * math.pi * math.sqrt((self.dy - self.d0) / (self.ay - self.a0))

    @property
    def Teq(self) -> float:
        """Effective period, FEMA 440 Eq. 3-5; meaningful only for (Sd, Sa) curves."""
        mu, a = self.mu, self.alpha
        if mu <= 1.0:
            return self.T0
        return self.T0 * math.sqrt(mu / (1 + a * mu - a))

    @property
    def d_array(self) -> FloatArray:
        return np.array([self.d0, self.dy, self.du])

    @property
    def a_array(self) -> FloatArray:
        return np.array([self.a0, self.ay, self.au])

    @property
    def is_admissible(self) -> bool:
        """True when the idealisation is physically meaningful."""
        values = (self.d0, self.dy, self.du, self.a0, self.ay, self.au)
        return all(math.isfinite(v) for v in values) and self.d0 < self.dy <= self.du and self.ay > self.a0

    def quantities(self) -> dict[str, float]:
        return {
            "x_0": self.d0,
            "x_y": self.dy,
            "x_u": self.du,
            "y_0": self.a0,
            "y_y": self.ay,
            "y_u": self.au,
            "k_el": self.kel,
            "k_inel": self.kinel,
            "μ": self.mu,
            "α": self.alpha,
        }
