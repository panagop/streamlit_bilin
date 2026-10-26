# bilin

Bilinear idealisation of force–deformation (capacity) curves by the iterative
equal-energy procedure (elastic stiffness = secant at 0.6·V_y), with a Streamlit
front end.

The algorithm is a self-contained port of `streng.tools.bilin` (streng 0.0.7),
which no longer imports on Python ≥ 3.11. It is regression-tested against the
original for all examples and option combinations
(`tests/reference_streng_0_0_7.json`).

## Run

```powershell
uv sync
uv run streamlit run app.py
```

## Use as a library

```python
from bilin import BilinOptions, Curve, bilinearize

result = bilinearize(Curve.from_file("examples/Example3"), BilinOptions(xtarget=0.35))
print(result.curve.quantities())
```

Input files: two numeric columns (x, y) separated by spaces, tabs, `;` or `,`.
Lines starting with `#` are ignored.

## Options

| option | default | meaning |
|---|---|---|
| `xtarget` | 0 | use the curve up to this deformation (0 = whole curve) |
| `dropstrength` | 0.75 | cut the curve at the first point after V_max with V ≤ 0.75·V_max |
| `elastoplastic` | False | force zero post-yield stiffness |
| `allowa010` | True | if False, cap the hardening ratio α at 0.10 |

Curves that end with an abrupt strength drop (e.g. `examples/Example6`, `Example7`)
give a non-physical result; `bilinearize` raises `BilinearizationError` instead of
returning it.

## Develop

```powershell
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy
```
