# simple module-level store
from dataclasses import dataclass

@dataclass
class ShieldInfo:
    unsafe: int = 0
    orig_idx: int = -1
    idx: int = -1
    orig_risk: float = 0.0
    chosen_risk: float = 0.0
    min_left: float = 0.0
    min_right: float = 0.0
    lam: float = 0.0   # added field for lambda value

_last = ShieldInfo()

def set(info: ShieldInfo):
    global _last
    _last = info

def get() -> ShieldInfo:
    return _last
