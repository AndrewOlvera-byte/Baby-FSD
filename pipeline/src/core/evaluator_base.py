from __future__ import annotations
from typing import Protocol, Dict, Any


class BaseEvaluator(Protocol):
    def __call__(self, model) -> Dict[str, Any]:
        ...


