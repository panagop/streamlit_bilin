"""Regenerate tests/reference_streng_0_0_7.json from the original streng implementation.

Not part of the test suite. streng 0.0.7 only imports on Python <= 3.10:

    uv run --no-project --python 3.10 --with streng==0.0.7 --with "numpy<2" \
        python tests/generate_reference.py examples > tests/reference_streng_0_0_7.json
"""

import json
import sys
from pathlib import Path

from streng.common.math.charts.linexy import LineXY
from streng.tools.bilin import Bilin

ex = Path(sys.argv[1])
cases = []
for name in [f"Example{i}" for i in range(1, 9)]:
    for xt in (0.0, 0.05, 0.35):
        for ep in (False, True):
            for a010 in (True, False):
                bl = Bilin(curve_ini=LineXY(), xtarget=xt, elastoplastic=ep, allowa010=a010)
                bl.curve_ini.load_delimited(str(ex / name), " ")
                bl.calc()
                c = bl.bilinear_curve
                cases.append(
                    dict(
                        file=name, xtarget=xt, elastoplastic=ep, allowa010=a010,
                        d0=float(c.d0), dy=float(c.dy), du=float(c.du),
                        a0=float(c.a0), ay=float(c.ay), au=float(c.au),
                        iterations=len(bl.output.outputTables["Iterations"].data),
                    )
                )
print(json.dumps(cases, indent=1))
