"""Measure cold startup and steady throughput using the actual isolated math runtime."""

import argparse
import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evalscope.metrics.math.contracts import MathRequest
from evalscope.metrics.math.runtime import MathWorkerPool


def measure(workers: int, count: int) -> dict:
    """Include cold startup separately; assert correctness and reap every measured child."""
    requests = [
        MathRequest(prediction=p, reference=g)
        for p, g in [
            ('18%', '.18'),
            (r'\frac{1}{2}', '.5'),
            ('x+x', '2x'),
            ('1800', '18'),
        ]
    ]
    with MathWorkerPool(max_workers=workers) as pool:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(pool.execute, [requests[0]] * workers))
        cold = time.perf_counter() - started
        latencies = []

        def score(index: int) -> bool:
            before = time.perf_counter()
            result = pool.execute(requests[index % len(requests)])
            latencies.append(time.perf_counter() - before)
            assert result.matched == (index % len(requests) != 3)
            return result.matched

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(score, range(count)))
        elapsed = time.perf_counter() - started
        children = [worker.process for worker in pool._workers.values()]
    assert all(child.poll() is not None for child in children)
    return {
        'workers': workers,
        'requests': count,
        'workload': 'four repeated simple expressions; upstream caches warm',
        'cold_seconds': cold,
        'steady_seconds': elapsed,
        'requests_per_second': count / elapsed,
        'median_seconds': statistics.median(latencies),
        'children_reaped': True,
    }


def main() -> None:
    """Write comparable measurements without model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('validation/math_verify/runtime.json'))
    args = parser.parse_args()
    results = {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'measurements': [measure(workers, args.count) for workers in [1, 4]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
