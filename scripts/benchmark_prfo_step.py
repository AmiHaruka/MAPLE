#!/usr/bin/env python3
"""Single-thread, dense-oracle P-RFO step microbenchmark; not a TS benchmark."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from maple.function.dispatcher.ts.algorithm.PRFO import prfo_step, to_numpy_f64, vec1d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='3145bf9c23c2e41600fc5734bcdb0a54c5a1dfca')
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error('--samples must be at least 3')
    source_path = 'maple/function/dispatcher/ts/algorithm/PRFO.py'
    source = subprocess.check_output(['git', 'show', f'{args.reference}:{source_path}'], text=True)
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'prfo_step')
    namespace = dict(np=np, to_numpy_f64=to_numpy_f64, vec1d=vec1d)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<dense-reference>', 'exec'), namespace)
    dense = namespace['prfo_step']
    rows = []
    for n in (30, 90, 300, 900):
        w = np.linspace(0.05, 2.0, n)
        w[0] = -0.5
        V, H = np.eye(n), np.diag(w)
        g = np.sin(np.arange(n) + 1) * 0.05
        kwargs = dict(is_ts=True, trust_radius=0.2, pre_eig=(w, V, g))
        timings = [[], []]
        for fn in (dense, prfo_step):
            fn(H, g, **kwargs)
        steps = []
        for sample in range(args.samples):
            # Alternate order to avoid assigning warm-cache bias to one solver.
            order = (0, 1) if sample % 2 == 0 else (1, 0)
            outputs = [None, None]
            for i in order:
                start = time.perf_counter()
                outputs[i] = (dense, prfo_step)[i](H, g, **kwargs)
                timings[i].append(time.perf_counter() - start)
            np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-10, atol=1e-12)
            steps.append(float(np.max(np.abs(outputs[0] - outputs[1]))))
        old, new = (statistics.median(t) for t in timings)
        row = dict(dofs=n, dense_seconds=old, secular_seconds=new,
                   speedup=old/new, max_abs_step_error=max(steps), samples=timings)
        rows.append(row)
        print({k: v for k, v in row.items() if k != 'samples'}, flush=True)
    sources = [source_path, 'maple/function/dispatcher/ts/algorithm/_rfo_arrowhead.py']
    record = dict(scope='synthetic_step_only_not_TS_accuracy_or_job_speed',
                  reference=args.reference, samples=args.samples, rows=rows,
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
                  gates=dict(width30_no_more_than_10pct_slowdown=rows[0]['speedup'] >= 1/1.1,
                             width300_at_least_2x=rows[2]['speedup'] >= 2))
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(record, indent=2) + '\n')
    return 0 if all(record['gates'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
