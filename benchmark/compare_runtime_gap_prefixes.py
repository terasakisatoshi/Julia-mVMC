"""Compare separately generated Julia prefixes; run through uv."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

before, after, summary_path = map(Path, sys.argv[1:])
# These are the existing 20-step runner budgets, not a new relaxed tolerance.
ABSOLUTE = RELATIVE = 1e-11
summary = dict(cases=0, rng_blocks=0, numeric_fields=0, max_absolute_difference=0.0,
               absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE, rng_sha256={})
for ranks, threads in ((1, 4), (2, 2), (4, 1)):
    layout = f"ranks{ranks}-threads{threads}"
    for size in (16, 24, 32):
        left, right = (root / layout / f"L{size}" for root in (before, after))
        for rank in range(ranks):
            for checkpoint in ("initial", "next"):
                name = f"rng-{checkpoint}-rank-{rank}.txt"
                a, b = ([int(word) for word in (root / name).read_text().split()]
                        for root in (left, right))
                assert len(a) == len(b) == 624
                assert all(0 <= word < 2**32 for word in a + b)
                assert a == b, f"SFMT block mismatch: {layout}/L{size}/{name}"
                summary["rng_blocks"] += 1
                summary["rng_sha256"][f"{layout}/L{size}/{name}"] = hashlib.sha256((left / name).read_bytes()).hexdigest()
        names = {path.name for path in left.glob("*.dat")}
        assert names == {path.name for path in right.glob("*.dat")}
        assert {"zvo_out.dat", "zqp_opt.dat"} <= names
        for name in sorted(names):
            rows = [(root / name).read_text().splitlines() for root in (left, right)]
            assert len(rows[0]) == len(rows[1]), (layout, size, name, "row count")
            for line, (a, b) in enumerate(zip(*rows), 1):
                aa, bb = a.split(), b.split()
                assert len(aa) == len(bb), (layout, size, name, line, "column count")
                for column, (x, y) in enumerate(zip(aa, bb), 1):
                    try:
                        xx, yy = float(x), float(y)
                    except ValueError:
                        assert x == y, (layout, size, name, line, column, "text")
                        continue
                    if re.fullmatch(r"[+-]?\d+", x) and re.fullmatch(r"[+-]?\d+", y):
                        assert int(x) == int(y), (layout, size, name, line, column, "integer")
                    else:
                        assert math.isfinite(xx) and math.isfinite(yy)
                        difference = abs(xx - yy)
                        assert difference <= ABSOLUTE + RELATIVE * max(abs(xx), abs(yy)), (
                            layout, size, name, line, column, xx, yy)
                        summary["numeric_fields"] += 1
                        summary["max_absolute_difference"] = max(summary["max_absolute_difference"], difference)
        summary["cases"] += 1
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "rng_sha256"}, indent=2))
