"""Validate captured-photo evidence without claiming machine-verifiable authenticity."""
from pathlib import Path
import cv2
from src.dataset import paired_files, indexed_files
from src.utils import read_json

CONDITIONS = {'indoor_outdoor', 'daylight_tubelight', 'near_far', 'clutter', 'occlusion'}


def verify_captured(root: str | Path) -> dict:
    root = Path(root)
    pairs = paired_files(root/'images_256', root/'masks_256')
    originals = indexed_files(root/'original')
    if pairs.keys() != originals.keys() or not 5 <= len(pairs) <= 10:
        raise ValueError('Captured set requires 5–10 matched originals, resized images and masks')
    meta = read_json(root/'metadata.json')
    if not meta.get('authentic_phone_photos') or not meta.get('manually_annotated'):
        raise ValueError('Author must attest genuine phone captures and manual annotation')
    records = meta.get('photos', [])
    if len(records) != len(pairs) or {r['id'] for r in records} != pairs.keys():
        raise ValueError('Every photograph needs one metadata record')
    covered = set()
    for r in records:
        if not r.get('object') or not r.get('annotation_method'):
            raise ValueError('Document object and annotation method for each photograph')
        covered.update(r.get('conditions', []))
    if not covered <= CONDITIONS or len(covered) < 3:
        raise ValueError('At least three documented condition categories are required')
    # Human review must confirm "both" categories actually cover each alternative.
    for image, mask in pairs.values():
        x = cv2.imread(str(image), cv2.IMREAD_COLOR)
        y = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if image.suffix.lower() != '.png' or mask.suffix.lower() != '.png':
            raise ValueError('Captured resized images and masks must be PNG')
        if x is None or y is None or x.shape != (256, 256, 3) or y.shape != (256, 256):
            raise ValueError('Captured images and masks must already be 256 square')
        if not bool(((y == 0) | (y == 255)).all()):
            raise ValueError('Captured annotation must use only 0 and 255')
    return dict(images=len(pairs), covered=sorted(covered), authenticity='author-attested; manual review required')
