from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class MetricSeries:
    values: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    def add(self, step: int, value: float):
        self.steps.append(int(step))
        self.values.append(float(value))


class ProgressLogger:
    def __init__(
        self,
        total:int,
        desc:str,
        outdir: Path,
        unit: str = "step",
        position: int = 0,
        leave: bool = True,
    ):
        self.total = int(total)
        self.start_time = time.time()
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.bar = tqdm(total=self.total, desc=desc, unit=unit, position=position, leave=leave)
        self.metrics: Dict[str, MetricSeries] = {}
        self.latest: Dict[str, float] = {}

    def update(self, n:int = 1, **display_metrics: float):
        if display_metrics:
            self.latest.update({k: float(v) for k, v in display_metrics.items()})
            self.bar.set_postfix({k: f"{v:.4f}" for k, v in self.latest.items()})
        self.bar.update(n)

    def add_metric_point(self, name: str, step: int, value: float):
        series = self.metrics.setdefault(name, MetricSeries())
        series.add(step, value)

    def close(self):
        self.bar.close()

    def elapsed_seconds(self) -> float:
        return float(time.time() - self.start_time)

    def samples_per_sec(self, processed: int) -> float:
        secs = self.elapsed_seconds()
        if secs <= 0:
            return 0.0
        return float(processed) / float(secs)

    def save_plot(self, filename: str = "graph.png", keys: Optional[List[str]] = None):
        if not self.metrics:
            return None
        keys = keys or list(self.metrics.keys())
        plt.figure(figsize=(10, 6))
        for key in keys:
            series = self.metrics.get(key)
            if series and series.steps:
                plt.plot(series.steps, series.values, label=key)
        plt.xlabel("step")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        out_path = self.outdir / filename
        plt.savefig(out_path)
        plt.close()
        return str(out_path)


