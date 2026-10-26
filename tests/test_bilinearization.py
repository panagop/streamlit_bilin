"""Regression tests against the original streng 0.0.7 implementation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bilin import BilinearCurve, BilinearizationError, BilinOptions, Curve, bilinearize

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
REFERENCE: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "reference_streng_0_0_7.json").read_text(encoding="utf-8")
)


def _admissible(case: dict[str, Any]) -> bool:
    return BilinearCurve(
        dy=case["dy"], ay=case["ay"], du=case["du"], au=case["au"], d0=case["d0"], a0=case["a0"]
    ).is_admissible


def _id(case: dict[str, Any]) -> str:
    return f"{case['file']}-xt{case['xtarget']}-ep{int(case['elastoplastic'])}-a010{int(case['allowa010'])}"


def _run(case: dict[str, Any]) -> BilinearCurve:
    options = BilinOptions(
        xtarget=case["xtarget"], elastoplastic=case["elastoplastic"], allowa010=case["allowa010"]
    )
    return bilinearize(Curve.from_file(EXAMPLES / case["file"]), options).curve


GOOD = [c for c in REFERENCE if _admissible(c)]
BAD = [c for c in REFERENCE if not _admissible(c)]


def test_reference_split() -> None:
    assert len(GOOD) == 80
    assert len(BAD) == 16


@pytest.mark.parametrize("case", GOOD, ids=_id)
def test_matches_streng(case: dict[str, Any]) -> None:
    bl = _run(case)
    for key in ("d0", "dy", "du", "a0", "ay", "au"):
        assert getattr(bl, key) == pytest.approx(case[key], rel=1e-9, abs=1e-12), key


@pytest.mark.parametrize("case", BAD, ids=_id)
def test_non_physical_results_raise(case: dict[str, Any]) -> None:
    """Abrupt strength drop at the end (Examples 6, 7) -> streng returned d_y < 0 or NaN."""
    with pytest.raises(BilinearizationError):
        _run(case)


def test_equal_energy() -> None:
    """Area under the bilinear curve equals the area under the (truncated) curve."""
    result = bilinearize(Curve.from_file(EXAMPLES / "Example1"))
    bl = result.curve
    bilinear_area = float(np.trapezoid(bl.a_array, bl.d_array - bl.d0))
    assert bilinear_area == pytest.approx(result.area, rel=1e-9)


def test_kel_is_secant_at_06_vy() -> None:
    result = bilinearize(Curve.from_file(EXAMPLES / "Example3"), BilinOptions(xtarget=0.35))
    bl = result.curve
    x = result.x_used - result.x_used[0]
    x06 = float(np.interp(0.6 * bl.ay, result.y_used, x))
    assert bl.kel == pytest.approx(0.6 * bl.ay / x06, rel=1e-5)


def test_initial_offset_only_shifts_curve() -> None:
    """Example2 is Example1 shifted by 0.005 in x."""
    a = bilinearize(Curve.from_file(EXAMPLES / "Example1")).curve
    b = bilinearize(Curve.from_file(EXAMPLES / "Example2")).curve
    assert b.d0 == pytest.approx(0.005)
    assert b.dy - b.d0 == pytest.approx(a.dy - a.d0)
    assert b.ay == pytest.approx(a.ay)


def test_alpha_cap() -> None:
    bl = _run({"file": "Example8", "xtarget": 0.0, "elastoplastic": False, "allowa010": False})
    assert bl.alpha <= 0.1 + 1e-9


def test_elastoplastic_has_zero_hardening() -> None:
    bl = _run({"file": "Example3", "xtarget": 0.35, "elastoplastic": True, "allowa010": True})
    assert bl.ay == pytest.approx(bl.au)


def test_derived_quantities() -> None:
    bl = BilinearCurve(dy=0.01, ay=100.0, du=0.05, au=120.0)
    assert bl.mu == pytest.approx(5.0)
    assert bl.kel == pytest.approx(1e4)
    assert bl.kinel == pytest.approx(500.0)
    assert bl.alpha == pytest.approx(0.05)
    expected_beta0 = (2 / math.pi) * (4 * 0.95) / (5 * (1 + 0.05 * 5 - 0.05))
    assert bl.beta0 == pytest.approx(expected_beta0)


@pytest.mark.parametrize("sep", [" ", "\t", ";", ","])
def test_parsing_delimiters(sep: str) -> None:
    curve = Curve.from_text(f"# header\n0{sep}0\n0.01{sep}50\n0.02{sep}60\n")
    np.testing.assert_allclose(curve.x, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(curve.y, [0.0, 50.0, 60.0])


def test_semicolon_csv_loads() -> None:
    csv = Curve.from_file(EXAMPLES / "Book1.csv")
    assert csv.x[0] == 0.0
    assert len(csv.x) > 3
