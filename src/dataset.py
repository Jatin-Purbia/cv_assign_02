"""Strict stem matching and immutable deterministic DUTS manifests."""
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.preprocessing import read_pair, image_tensor

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def indexed_files(folder: str | Path) -> dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing data directory: {folder}')
    result = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError(f'Duplicate sample ID: {path.stem}')
            result[path.stem] = path
    return result


def paired_files(images: str | Path, masks: str | Path) -> dict[str, tuple[Path, Path]]:
    photos, labels = indexed_files(images), indexed_files(masks)
    if photos.keys() != labels.keys():
        raise ValueError(f'Unmatched images/masks: {sorted(photos.keys() ^ labels.keys())[:10]}')
    if not photos:
        raise ValueError('Empty dataset')
    return {key: (photos[key], labels[key]) for key in photos}


def read_ids(path: str | Path) -> list[str]:
    ids = Path(path).read_text(encoding='utf-8').splitlines()
    if not ids or any(not value or value != value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f'Empty, duplicate or malformed manifest: {path}')
    return ids


def split_ids(ids: list[str], seed: int = 42, fraction: float = .1) -> tuple[list[str], list[str]]:
    if len(ids) != len(set(ids)) or len(ids) < 2 or not 0 < fraction < 1:
        raise ValueError('Invalid split input')
    values = sorted(ids)
    random.Random(seed).shuffle(values)
    nval = max(1, min(len(ids)-1, round(len(ids)*fraction)))
    return sorted(values[nval:]), sorted(values[:nval])


class SegmentationDataset(Dataset):
    def __init__(self, images: str | Path, masks: str | Path,
                 ids: list[str] | None = None, augment: bool = False):
        self.pairs = paired_files(images, masks)
        self.ids = sorted(self.pairs) if ids is None else ids
        if not self.ids or len(set(self.ids)) != len(self.ids) or not set(self.ids) <= self.pairs.keys():
            raise ValueError('Manifest is empty, duplicated or contains unknown IDs')
        self.augment = augment
        self._cache = {}  # decoded 256x256 pairs; avoids re-decoding JPEGs every epoch

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.ids[index]
        if key not in self._cache:
            self._cache[key] = read_pair(*self.pairs[key])
        photo, label = self._cache[key]
        x = image_tensor(photo)
        y = torch.from_numpy(label).float().unsqueeze(0)
        if self.augment:
            if torch.rand(()).item() < .5:
                x, y = x.flip(-1), y.flip(-1)
            brightness = .9 + .2*torch.rand(()).item()
            contrast = .9 + .2*torch.rand(()).item()
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = ((x-mean)*contrast+mean).mul(brightness).clamp(0, 1)
        return x, y
