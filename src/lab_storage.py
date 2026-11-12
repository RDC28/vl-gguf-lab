"""Incremental local/Drive run storage with provenance-checked resumption."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def results_root(drive_root='MyDrive/vl-gguf-lab/results', local_root=None):
    try:
        from google.colab import drive
    except ImportError:
        return Path(local_root) if local_root else ROOT / 'results'
    drive.mount('/content/drive')
    relative = Path(drive_root)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('drive_root must be relative to /content/drive')
    return Path('/content/drive') / relative


class Run:
    def __init__(self, case, config, *, root=None, resume=None):
        root = Path(root) if root else results_root(config.get('drive_root', 'MyDrive/vl-gguf-lab/results'))
        parent = root / f'{case:02d}' / 'runs'
        parent.mkdir(parents=True, exist_ok=True)
        if resume:
            self.path = parent / resume
            if Path(resume).name != resume or not self.path.is_dir():
                raise ValueError('resume must name an existing run in this case')
            saved = json.loads((self.path / 'config.json').read_text())
            if saved != config:
                raise ValueError('Resume configuration differs; start a new run')
        else:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            self.path = parent / f'{stamp}-{uuid.uuid4().hex[:8]}'
            self.path.mkdir()
            atomic_json(self.path / 'config.json', config)
            packages = {}
            for name in ('torch', 'transformers', 'llama-cpp-python', 'gguf', 'numpy', 'peft'):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    pass
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            atomic_json(self.path / 'environment.json', dict(python=sys.version, platform=platform.platform(),
                                                            packages=packages, commit=commit))
        self.console = sys.stdout
        self.log('Run opened: ' + str(self.path))

    def json(self, name, value):
        atomic_json(self.path / name, value)

    def rows(self, name):
        path = self.path / name
        if not path.exists():
            return []
        rows = []
        # A disconnected append may leave an incomplete final line. Discard only that tail.
        with path.open('rb+') as stream:
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    stream.truncate(start)
                    break
                rows.append(json.loads(line))
        return rows

    def append(self, name, value):
        with (self.path / name).open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def log(self, message):
        self.append('events.jsonl', dict(time=datetime.now(timezone.utc).isoformat(), message=message))
        print(message, file=self.console, flush=True)

    @contextmanager
    def capture(self):
        # Library and training diagnostics go directly to the persistent run directory.
        with (self.path / 'runtime.log').open('a', buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                yield
