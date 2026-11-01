"""One-image-at-a-time quality and CPU-process timing, with file I/O excluded."""
from datetime import datetime, timezone
from pathlib import Path
import torch
from src.dataset import paired_files, read_ids
from src.inference import verify_frozen
from src.optimized_inference import ConfiguredPredictor as Predictor
from src.metrics import MetricAccumulator, ces
from src.preprocessing import read_image, read_pair
from src.utils import environment, sha256, write_json


def evaluate(weights: str | Path, images: str | Path, masks: str | Path,
             output: str | Path, split: str, manifest: str | Path | None = None) -> dict:
    if split not in ('validation', 'duts-test', 'captured', 'synthetic'):
        raise ValueError('Unknown evaluation split')
    output = Path(output)
    if split == 'duts-test' and output.exists():
        raise FileExistsError('Final-test result exists; do not retune/repeat against DUTS-TE')
    frozen = verify_frozen(weights) if split in ('duts-test', 'captured') else None
    pairs = paired_files(images, masks)
    ids = read_ids(manifest) if manifest else sorted(pairs)
    if split == 'validation' and manifest is None:
        raise ValueError('Validation requires the saved validation manifest')
    if not set(ids) <= pairs.keys():
        raise ValueError('Manifest contains unknown sample IDs')
    if split == 'duts-test' and (len(ids) != 5019 or set(ids) != pairs.keys()):
        raise ValueError('Official test requires all 5,019 pairs without a subset manifest')
    if split == 'captured':
        from src.captured import verify_captured
        verify_captured(Path(images).parent)
        if set(ids) != pairs.keys():
            raise ValueError('Captured evaluation must include every photograph')
    predictor, metrics = Predictor(weights), MetricAccumulator()
    predictor(read_image(pairs[ids[0]][0]))  # warmup excluded
    total_ms, per_image = 0.0, []
    for key in ids:
        image, target = read_pair(*pairs[key])
        prediction, elapsed = predictor.timed(image)
        previous = (metrics.tn, metrics.fp, metrics.fn, metrics.tp)
        diagnostic = metrics.update(torch.from_numpy(prediction), torch.from_numpy(target))
        counts = [value-old for value, old in zip((metrics.tn, metrics.fp, metrics.fn, metrics.tp), previous)]
        diagnostic['confusion_matrix'] = [counts[:2], counts[2:]]
        total_ms += elapsed
        per_image.append(dict(id=key, cpu_ms=elapsed, **diagnostic))
    result = dict(**metrics.compute(), cpu_ms=total_ms/len(ids), total_cpu_ms=total_ms,
                  size_bytes=predictor.size_bytes, checkpoint_sha256=sha256(weights),
                  checkpoint=str(weights), split=split, timing='time.process_time; full pipeline; warmup excluded',
                  evaluated_utc=datetime.now(timezone.utc).isoformat(), environment=environment(),
                  manifest_sha256=sha256(manifest) if manifest else None,
                  official=split == 'duts-test', frozen=frozen, per_image=per_image)
    result['memory_format'] = predictor.memory_format
    result['ces'] = ces(result['q'], result['cpu_ms'], result['size_bytes'])
    write_json(output, result)
    return result
