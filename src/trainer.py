"""Reproducible training, validation and best-Q checkpoint selection."""
import csv
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.dataset import SegmentationDataset, read_ids
from src.inference import export_weights
from src.losses import segmentation_loss
from src.metrics import MetricAccumulator
from src.models import build_model
from src.utils import environment, seed_everything, sha256, write_json


def train(config: dict, output: str | Path, device: str = 'cpu') -> dict:
    output = Path(output)
    if (output/'run.json').exists() or (output/'history.csv').exists():
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
    training = SegmentationDataset(*pairs, trids, augment=True)
    validation = SegmentationDataset(*pairs, valids)
    if set(trids) | set(valids) != set(training.pairs):
        raise ValueError('Manifests must partition all available DUTS-TR IDs')
    if not config.get('synthetic_smoke', False) and len(training.pairs) != 10553:
        raise ValueError('Real training requires all 10,553 DUTS-TR pairs')
    loader = DataLoader(training, batch_size=config['batch_size'], shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(config['seed']))
    val_loader = DataLoader(validation, batch_size=config['batch_size'], num_workers=0)
    model = build_model(config['model']).to(device)
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
                      parameters=sum(p.numel() for p in model.parameters()))
    write_json(output/'run.json', provenance)
    best_q, best_epoch, stale = -1.0, 0, 0
    columns = ['epoch', 'train_loss', 'val_loss', 'miou', 'bf1', 'q', 'lr', 'seconds']
    for epoch in range(1, config['epochs']+1):
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
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
        with (output/'history.csv').open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"Epoch {epoch}: train={row['train_loss']:.5f} val Q={scores['q']:.5f}", flush=True)
        if scores['q'] > best_q:
            best_q, best_epoch, stale = scores['q'], epoch, 0
            export_weights(output/'best.pth', model, config['model'],
                           int(config.get('inference_resolution', resolution)),
                           float(config.get('threshold', .5)))
            write_json(output/'best.json', dict(epoch=epoch, **scores))
        else:
            stale += 1
        scheduler.step()
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), seed=config['seed']), output/'last_checkpoint.pth')
        if stale >= config['patience']:
            break
    summary = dict(best_epoch=best_epoch, best_q=best_q, completed_epochs=epoch,
                   synthetic_smoke=config.get('synthetic_smoke', False))
    write_json(output/'summary.json', summary)
    return summary
