"""Reproducible training, validation and best-Q checkpoint selection."""
import csv
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.dataset import SegmentationDataset, read_ids
from src.inference import export_weights
from src.losses import segmentation_loss
from src.metrics import MetricAccumulator
from src.models import build_model
from src.utils import ROOT, environment, seed_everything, sha256, write_json, read_json


def _save_atomic(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _write_history(path: Path, rows: list[dict], columns: list[str]) -> None:
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(config: dict, output: str | Path, device: str = 'cpu',
          resume: bool = False, max_epochs: int | None = None) -> dict:
    output = Path(output)
    if max_epochs is not None and max_epochs < 1:
        raise ValueError('max_epochs must be positive')
    if not resume and ((output/'run.json').exists() or (output/'history.csv').exists()):
        raise FileExistsError('Choose a new run directory; existing history must not be overwritten')
    seed_everything(int(config['seed']))
    training_threads = int(config.get('training_threads', 1))
    if not 1 <= training_threads <= 16:
        raise ValueError('training_threads must be between 1 and 16')
    # Only training may use additional CPU threads. Evaluation entry points
    # always configure a fresh single-thread process before numerical imports.
    torch.set_num_threads(training_threads)
    root = Path(config['data_root'])
    trids, valids = read_ids(config['train_manifest']), read_ids(config['val_manifest'])
    if set(trids) & set(valids):
        raise ValueError('Training/validation overlap')
    pairs = (root/'DUTS-TR'/'DUTS-TR-Image', root/'DUTS-TR'/'DUTS-TR-Mask')
    training = SegmentationDataset(*pairs, trids, augment=True,
                                   augmentation=config.get('augmentation'))
    validation = SegmentationDataset(*pairs, valids)
    if set(trids) | set(valids) != set(training.pairs):
        raise ValueError('Manifests must partition all available DUTS-TR IDs')
    if not config.get('synthetic_smoke', False) and len(training.pairs) != 10553:
        raise ValueError('Real training requires all 10,553 DUTS-TR pairs')
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(training, batch_size=config['batch_size'], shuffle=True,
                        num_workers=0, generator=generator)
    val_loader = DataLoader(validation, batch_size=config['batch_size'], num_workers=0)
    model = build_model(config['model']).to(device)
    initialization = None
    if config.get('initialize_from'):
        initial_path = Path(config['initialize_from'])
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        if initial['model'] != config['model']:
            raise ValueError('Initialization architecture does not match training configuration')
        model.load_state_dict(initial['state_dict'], strict=True)
        initialization = dict(path=initial_path.as_posix(), sha256=sha256(initial_path),
                              method='Fine-tune own scratch-trained model; reset optimizer and scheduler')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    resolution = int(config['resolution'])
    if not 32 <= resolution <= 256 or config['epochs'] < 1 or config['patience'] < 1:
        raise ValueError('Invalid training configuration')
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(config=config, environment=environment(), device=device,
                      train_count=len(training), val_count=len(validation),
                      train_manifest_sha256=sha256(config['train_manifest']),
                      val_manifest_sha256=sha256(config['val_manifest']),
                      parameters=sum(p.numel() for p in model.parameters()), initialization=initialization,
                      source_sha256={f: sha256(ROOT/f) for f in
                          ('src/trainer.py', 'src/dataset.py', 'src/losses.py', 'src/models.py')})
    best_q, best_epoch, stale = -1.0, 0, 0
    columns = ['epoch', 'train_loss', 'val_loss', 'miou', 'bf1', 'q', 'lr', 'seconds']
    start_epoch, history, best_state, best_scores = 1, [], None, None
    if resume:
        previous = read_json(output/'run.json')
        for key in ('config', 'device', 'train_manifest_sha256', 'val_manifest_sha256',
                    'source_sha256', 'initialization'):
            if previous.get(key) != provenance[key]:
                raise ValueError(f'Resume provenance mismatch: {key}')
        saved = torch.load(output/'last_checkpoint.pth', map_location=device, weights_only=True)
        if saved.get('format_version') != 2:
            raise ValueError('Old checkpoints lack RNG state; use initialize_from for a new fine-tuning run')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['loader_rng'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
        random.setstate(saved['python_rng'])
        state = saved['numpy_rng']
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        best_q, best_epoch, stale = saved['best_q'], saved['best_epoch'], saved['stale']
        history, best_state, best_scores = saved['history'], saved['best_state'], saved['best_scores']
        start_epoch = saved['epoch'] + 1
        _write_history(output/'history.csv', history, columns)
        if best_state is not None:
            restored = build_model(config['model'])
            restored.load_state_dict(best_state)
            export_weights(output/'best.pth', restored, config['model'],
                           int(config.get('inference_resolution', resolution)),
                           float(config.get('threshold', .5)))
            write_json(output/'best.json', dict(epoch=best_epoch, **best_scores))
            # Constructing the recovery model consumed RNG; restore it again.
            torch.set_rng_state(saved['torch_rng'].cpu())
    else:
        write_json(output/'run.json', provenance)
    stop_epoch = min(config['epochs'], start_epoch + max_epochs-1) if max_epochs else config['epochs']
    epoch = start_epoch-1
    for epoch in range(start_epoch, stop_epoch+1):
        if stale >= config['patience']:
            break
        start = time.perf_counter()
        model.train()
        total_loss = 0.0
        for batch_index, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if resolution != 256:
                x = F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.shape[-2:] != (256, 256):
                logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
            loss = segmentation_loss(logits, y, config['dice_weight'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch}, batch {batch_index}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach())*len(x)
            if (batch_index+1) % 100 == 0:
                print(f'Epoch {epoch}: batch {batch_index+1}/{len(loader)}', flush=True)
        model.eval()
        val_loss, metrics = 0.0, MetricAccumulator()
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if resolution != 256:
                    x = F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)
                logits = model(x)
                if logits.shape[-2:] != (256, 256):
                    logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
                val_loss += float(segmentation_loss(logits, y, config['dice_weight']))*len(x)
                predictions = (logits.sigmoid() >= .5).cpu()
                for p, t in zip(predictions, y.cpu()):
                    metrics.update(p[0], t[0])
        scores = metrics.compute()
        row = dict(epoch=epoch, train_loss=total_loss/len(training), val_loss=val_loss/len(validation),
                   miou=scores['miou'], bf1=scores['bf1'], q=scores['q'],
                   lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-start)
        history.append(row)
        print(f"Epoch {epoch}: train={row['train_loss']:.5f} val Q={scores['q']:.5f}", flush=True)
        if scores['q'] > best_q:
            best_q, best_epoch, stale = scores['q'], epoch, 0
            best_state, best_scores = copy.deepcopy(model.state_dict()), dict(scores)
            export_weights(output/'best.pth', model, config['model'],
                           int(config.get('inference_resolution', resolution)),
                           float(config.get('threshold', .5)))
            write_json(output/'best.json', dict(epoch=epoch, **scores))
        else:
            stale += 1
        scheduler.step()
        numpy_rng = np.random.get_state()
        _save_atomic(dict(format_version=2, epoch=epoch, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         seed=config['seed'], torch_rng=torch.get_rng_state(),
                         loader_rng=generator.get_state(), python_rng=random.getstate(),
                         numpy_rng=(numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                         best_q=best_q, best_epoch=best_epoch, stale=stale,
                         best_state=best_state, best_scores=best_scores, history=history),
                     output/'last_checkpoint.pth')
        _write_history(output/'history.csv', history, columns)
        if stale >= config['patience']:
            break
    completed = len(history)
    finished = completed >= config['epochs'] or stale >= config['patience']
    summary = dict(best_epoch=best_epoch, best_q=best_q, completed_epochs=completed,
                   finished=finished,
                   synthetic_smoke=config.get('synthetic_smoke', False))
    write_json(output/('summary.json' if finished else 'progress.json'), summary)
    return summary
