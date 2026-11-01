"""Create explicitly prompted EfficientSAM-Ti reference masks; no ground truth."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import cv2
import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def overlay(photo, mask):
    result = photo.copy()
    result[mask] = (.6*photo[mask] + .4*np.array([0, 255, 0])).astype(np.uint8)
    return result


def panel(image, title):
    header = np.full((32, 256, 3), 255, np.uint8)
    cv2.putText(header, title, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', default='captured/original')
    parser.add_argument('--prompts', default='reference_masks/prompts.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh reference output directory')
    model_root = Path('tmp/reference_models/efficient_sam_ti')
    sys.path.insert(0, str(model_root.resolve()))
    from efficient_sam.efficient_sam import build_efficient_sam
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    model = build_efficient_sam(192, 3)
    checkpoint = torch.load(model_root/'weights/efficient_sam_vitt.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    prompts = json.loads(Path(args.prompts).read_text(encoding='utf-8'))
    photos = {p.stem: p for p in Path(args.images).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    if set(photos) != set(prompts['photos']):
        raise ValueError('Prompt records must cover exactly the source photographs')
    for name in ('masks_256', 'overlays', 'cutouts', 'individual_masks'):
        (output/name).mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    with torch.inference_mode():
        for index, (key, source) in enumerate(sorted(photos.items()), 1):
            started = time.perf_counter()
            photo = cv2.resize(cv2.imread(str(source)), (256, 256), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(photo[:, :, ::-1].copy()).permute(2, 0, 1).float()[None]/255
            # Each query selects one object; the saved reference is their union.
            points, labels = [], []
            for query in prompts['photos'][key]['queries']:
                x0, y0, x1, y1 = query['box']
                p = [[x0, y0], [x1, y1]] + query.get('positive', [])
                lab = [2, 3] + [1]*len(query.get('positive', []))
                if len(p) > 6:
                    raise ValueError('EfficientSAM accepts at most six prompt points per query')
                p += [[-1, -1]]*(6-len(p))
                lab += [-1]*(6-len(lab))
                points.append(p)
                labels.append(lab)
            logits, scores = model(x, torch.tensor([points], dtype=torch.float32), torch.tensor([labels]))
            mask = np.zeros((256, 256), dtype=bool)
            predictions = []
            for query_index in range(len(points)):
                candidate_index = int(scores[0, query_index].argmax())
                individual = (logits[0, query_index, candidate_index] >= 0).numpy()
                mask |= individual
                save(output/'individual_masks'/f'{key}_{query_index}.png', individual.astype(np.uint8)*255)
                predictions.append(dict(query=query_index, candidate=candidate_index,
                    predicted_iou=float(scores[0, query_index, candidate_index]),
                    foreground_fraction=float(individual.mean())))
            hard = mask.astype(np.uint8)*255
            save(output/'masks_256'/f'{key}.png', hard)
            composite = overlay(photo, mask)
            save(output/'overlays'/f'{key}.png', composite)
            save(output/'cutouts'/f'{key}.png', np.dstack([photo, hard]))
            prompt_view = photo.copy()
            for query in prompts['photos'][key]['queries']:
                x0, y0, x1, y1 = query['box']
                cv2.rectangle(prompt_view, (x0, y0), (x1, y1), (255, 100, 0), 1)
                for point in query.get('positive', []):
                    cv2.circle(prompt_view, tuple(point), 2, (0, 0, 255), -1)
            rows.append(np.concatenate([panel(prompt_view, key), panel(composite, 'EfficientSAM AI reference'),
                panel(cv2.cvtColor(hard, cv2.COLOR_GRAY2BGR), 'AI mask - not ground truth')], axis=1))
            records.append(dict(id=key, input_sha256=digest(source), target=prompts['photos'][key]['target'],
                queries=prompts['photos'][key]['queries'], candidate_selection=predictions,
                foreground_fraction=float(mask.mean()), mask_sha256=digest(output/'masks_256'/f'{key}.png')))
            print(f'{index}/{len(photos)} {key}: {mask.mean():.1%} foreground ({time.perf_counter()-started:.1f}s)', flush=True)
    save(output/'comparison.png', np.concatenate(rows, axis=0))
    metadata = dict(kind='AI-generated, assistant-prompted pretrained reference; not manual ground truth',
        model='EfficientSAM-Ti', download=json.loads((model_root/'download_provenance.json').read_text()),
        created_utc=datetime.now(timezone.utc).isoformat(), prompts_sha256=digest(args.prompts),
        selection='Highest model-predicted IoU per query; threshold logits >=0; union queries; no morphology',
        human_reviewed=False, allowed_as_assignment_ground_truth=False,
        note='Predicted IoU is a model estimate, not measured accuracy. Foreground definitions are assistant assumptions.',
        images=records)
    (output/'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved all prompted reference masks to {output}', flush=True)


if __name__ == '__main__':
    main()
