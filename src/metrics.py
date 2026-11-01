"""Assignment-specific metrics, implemented from tensor/array operations."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def _mask(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim != 2 or value.numel() == 0:
        raise ValueError('Expected nonempty 2D binary mask')
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError('Metric masks must contain only 0 and 1')
    return value.bool()


def boundary(mask: torch.Tensor) -> torch.Tensor:
    """Foreground with any four-connected background neighbor; outside is BG."""
    m = _mask(mask)
    pad = F.pad(m, (1, 1, 1, 1), value=False)
    interior = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return m & ~interior


def boundary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape != target.shape:
        raise ValueError('Mask shapes differ')
    p, t = boundary(prediction), boundary(target)
    npred, ntrue = int(p.sum()), int(t.sum())
    if not npred and not ntrue:
        return 1.0
    if not npred or not ntrue:
        return 0.0
    precision = float((p & _nearby(t)).sum()) / npred
    recall = float((t & _nearby(p)).sum()) / ntrue
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0


def _nearby(b: torch.Tensor) -> torch.Tensor:
    """Exact centered 5x5 match by 25 whole-array boolean ORs, no pixel loops.

    CPU boolean arrays avoid converting to float and repeatedly comparing
    25 floating values for every pixel. Outside the image remains false.
    """
    value = b.detach().cpu().numpy()
    height, width = value.shape
    padded = np.zeros((height+4, width+4), dtype=bool)
    padded[2:-2, 2:-2] = value
    matched = np.zeros_like(value)
    for dy in range(5):
        for dx in range(5):
            matched |= padded[dy:dy+height, dx:dx+width]
    return torch.from_numpy(matched).to(b.device)


def quality(miou: float, bf1: float) -> float:
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (miou, bf1)):
        raise ValueError('Metrics must be finite in [0,1]')
    return .7*miou + .3*bf1


def ces(q: float, cpu_ms: float, size_bytes: int) -> float:
    if not math.isfinite(q) or not 0 <= q <= 1 or not math.isfinite(cpu_ms) or cpu_ms < 0 or size_bytes < 0:
        raise ValueError('Invalid score inputs')
    return 0.0 if cpu_ms > 50 else q*(1-.2*cpu_ms/50-.1*min(1, size_bytes/4194304))


class MetricAccumulator:
    """One global matrix, rows=true [BG,FG], columns=prediction [BG,FG]."""
    def __init__(self):
        self.tn = self.fp = self.fn = self.tp = self.count = 0
        self.bf1_sum = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> dict:
        p, t = _mask(prediction), _mask(target)
        if p.shape != t.shape:
            raise ValueError('Mask shapes differ')
        tp, tn = int((p & t).sum()), int((~p & ~t).sum())
        fp, fn = int((p & ~t).sum()), int((~p & t).sum())
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        bf1 = boundary_f1(p, t)
        self.bf1_sum += bf1
        self.count += 1
        # Per-image diagnostics support genuine failure review; these IoUs are
        # NEVER averaged to obtain the dataset mIoU.
        fg = tp/(tp+fp+fn) if tp+fp+fn else 1.0
        bg = tn/(tn+fp+fn) if tn+fp+fn else 1.0
        return dict(miou=(fg+bg)/2, bf1=bf1, q=quality((fg+bg)/2, bf1))

    def compute(self) -> dict:
        if not self.count:
            raise ValueError('Cannot evaluate an empty dataset')
        error = self.fp + self.fn
        fg = self.tp/(self.tp+error) if self.tp+error else 1.0
        bg = self.tn/(self.tn+error) if self.tn+error else 1.0
        miou, bf1 = (fg+bg)/2, self.bf1_sum/self.count
        return dict(confusion_matrix=[[self.tn, self.fp], [self.fn, self.tp]],
                    iou_fg=fg, iou_bg=bg, miou=miou, bf1=bf1,
                    q=quality(miou, bf1), images=self.count)
