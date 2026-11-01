"""Workspace fixtures avoiding restrictive Windows mkdtemp ACLs in the sandbox."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid


@contextmanager
def temporary_directory(dir: Path):
    base = dir.resolve()
    target = base/uuid.uuid4().hex
    target.mkdir(parents=True)
    try:
        yield str(target)
    finally:
        if target.resolve().is_relative_to(base):
            shutil.rmtree(target)


class TemporaryDirectory:
    def __init__(self, dir: Path):
        self.context = temporary_directory(dir)
        self.name = self.context.__enter__()

    def cleanup(self):
        self.context.__exit__(None, None, None)
