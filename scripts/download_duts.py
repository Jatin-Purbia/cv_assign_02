"""Download authentic archives using stdlib; hashes are observations, not authority checksums."""
from src import runtime
runtime.configure()
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path
from src.utils import sha256, write_json

BASE = 'https://saliencydetection.net/duts/download/'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['TR', 'TE'], default='TR')
    p.add_argument('--out', default='data/DUTS')
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = f'DUTS-{args.split}.zip'
    archive = out/name
    if not archive.exists():
        temporary = archive.with_suffix('.partial')
        with urllib.request.urlopen(BASE+name, timeout=60) as source, temporary.open('wb') as dest:
            shutil.copyfileobj(source, dest)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (out/member.filename).resolve()
            if not target.is_relative_to(out):
                raise ValueError('Unsafe archive path')
        bundle.extractall(out)
    write_json(out/f'DUTS-{args.split}-source.json', dict(url=BASE+name, sha256=sha256(archive),
               bytes=archive.stat().st_size, note='Observed checksum; validate official counts and pairings separately'))


if __name__ == '__main__':
    main()
