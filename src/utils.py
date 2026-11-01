"""Reproducibility, provenance, and atomic result serialization."""
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, value: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    import cv2
    import matplotlib
    cpu = ''
    # Read-only registry query; no child process or WMI service needed.
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    if not cpu and Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    return dict(python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__,
                cpu_model=cpu or 'unavailable', platform=platform.platform(),
                torch_threads=torch.get_num_threads())
