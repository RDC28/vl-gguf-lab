"""Acquire one pinned small GGUF and inventory its immutable tensor bytes."""
from collections import Counter
import hashlib
import importlib.metadata
import json
import resource
import time
import urllib.request

from baseline import ROOT, check_memory, memory_state

REPO = 'bartowski/SmolLM2-135M-Instruct-GGUF'
REVISION = '09816acd5d99df7be770d85ea30822623dab342c'
NAME = 'SmolLM2-135M-Instruct-f16.gguf'
SIZE = 270885952
SHA256 = 'f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    result_path = ROOT / 'results/case3-source.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    start = time.perf_counter()
    before = check_memory()
    path = ROOT / 'models' / NAME
    if not path.exists():
        partial = path.with_suffix('.gguf.part')
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}'
        with urllib.request.urlopen(url, timeout=60) as source, partial.open('xb') as dest:
            while chunk := source.read(4 * 1024 * 1024):
                check_memory()
                dest.write(chunk)
        if partial.stat().st_size != SIZE or digest(partial) != SHA256:
            raise ValueError('Downloaded artifact does not match upstream size/hash')
        partial.rename(path)
    assert path.stat().st_size == SIZE and digest(path) == SHA256
    check_memory()
    from gguf import GGUFReader
    reader = GGUFReader(path, 'r')
    tensors = [{'name': t.name, 'shape': t.shape.tolist(), 'type': t.tensor_type.name,
                'elements': int(t.n_elements), 'bytes': int(t.n_bytes),
                'sha256': hashlib.sha256(t.data).hexdigest()} for t in reader.tensors]
    fields = {}
    for name, field in reader.fields.items():
        value = field.contents()
        fields[name] = {'types': [t.name for t in field.types],
                        'value_sha256': hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()}
        if not isinstance(value, list) or len(value) < 100:
            fields[name]['value'] = value
        else:
            fields[name]['count'] = len(value)
    target = next(t for t in reader.tensors if t.name == 'blk.0.attn_norm.weight')
    assert target.tensor_type.name == 'F32' and target.shape.tolist() == [576]
    result = {'repo': REPO, 'revision': REVISION, 'name': NAME, 'size_bytes': SIZE,
              'sha256': SHA256, 'gguf_version': importlib.metadata.version('gguf'),
              'fields': fields, 'tensors': tensors,
              'tensor_types': dict(Counter(t['type'] for t in tensors)),
              'stored_elements': sum(t['elements'] for t in tensors),
              'proposed_edit': {'tensor': target.name, 'operation': 'multiply by 1.01 in float32',
                                'minimum': float(target.data.min()), 'maximum': float(target.data.max()),
                                'control': 'No-change GGUFWriter rewrite first; no edits performed yet'},
              'memory_before': before, 'memory_after': memory_state(),
              'elapsed_seconds': time.perf_counter() - start,
              'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limitations': 'Acquisition and read-only mmap inventory only. No inference, rewrite or edit. RSS excludes some unified-memory allocations. GGUF conversion is distinct from case 2 BF16 source; no weight equivalence assumed.'}
    with result_path.open('x') as dest:
        json.dump(result, dest, indent=2)
        dest.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('fields', 'tensors')}, indent=2))


if __name__ == '__main__':
    main()
