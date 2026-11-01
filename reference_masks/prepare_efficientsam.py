"""Download the official EfficientSAM-Ti reference implementation and checkpoint."""
from pathlib import Path
import hashlib
import json
import urllib.request

root = Path('tmp/reference_models/efficient_sam_ti')
root.mkdir(parents=True, exist_ok=True)
# Match the exact model/source revision recorded with the captured reference masks.
provenance = json.loads(Path('captured/mask_provenance.json').read_text(encoding='utf-8'))
revision = provenance['download']['revision']
expected = {entry['path']: entry['sha256'] for entry in provenance['download']['files']}
base = f'https://raw.githubusercontent.com/yformer/EfficientSAM/{revision}/'
files = ['LICENSE', 'efficient_sam/efficient_sam.py', 'efficient_sam/efficient_sam_encoder.py',
         'efficient_sam/efficient_sam_decoder.py', 'efficient_sam/two_way_transformer.py',
         'efficient_sam/mlp.py', 'weights/efficient_sam_vitt.pt']
records = []
for relative in files:
    dest = root/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected[relative]:
        records.append(dict(path=relative, sha256=expected[relative]))
        continue
    temporary = dest.with_suffix(dest.suffix+'.download')
    print(f'Downloading official EfficientSAM: {relative}', flush=True)
    with urllib.request.urlopen(base+relative, timeout=90) as response, temporary.open('wb') as handle:
        while chunk := response.read(1024*1024):
            handle.write(chunk)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected[relative]:
        raise ValueError(f'Downloaded reference file checksum mismatch: {relative}')
    temporary.replace(dest)
    records.append(dict(path=relative, sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(root/'efficient_sam/__init__.py').touch()
(root/'download_provenance.json').write_text(json.dumps(dict(repository='https://github.com/yformer/EfficientSAM',
    revision=revision, files=records), indent=2), encoding='utf-8')
print('Downloaded source and weights; ready for inspection and local reference inference.', flush=True)
