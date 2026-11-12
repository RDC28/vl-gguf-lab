"""Audited F16/F32 Llama GGUF edits for the small controlled experiments."""
import hashlib
import re
from pathlib import Path

import numpy as np

from .lab_storage import digest


def rewrite(source, output, *, operation, layer=0, strength=1.0):
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    source, output = Path(source), Path(output)
    reader = GGUFReader(source, 'r')
    if reader.fields['general.architecture'].contents() != 'llama':
        raise ValueError('Edits support the small Llama-family control, not arbitrary VL architectures')
    count = reader.fields['llama.block_count'].contents()
    if not 0 <= layer < count:
        raise ValueError('Layer outside model')
    if operation not in ('norm', 'add', 'remove', 'control'):
        raise ValueError('Unknown edit')
    if any(t.tensor_type.name not in ('F16', 'F32') for t in reader.tensors):
        raise ValueError('Only unquantized F16/F32 edits are supported')
    mapping = list(range(count))
    if operation == 'add':
        mapping.insert(layer + 1, layer)
    elif operation == 'remove':
        mapping.pop(layer)
    tensors = {t.name: t for t in reader.tensors}
    plan = {}
    for name in tensors:
        if not name.startswith('blk.'):
            plan[name] = name
    for dest, origin in enumerate(mapping):
        for name in tensors:
            if name.startswith(f'blk.{origin}.'):
                plan[re.sub(r'^blk\.\d+\.', f'blk.{dest}.', name)] = name
    writer = GGUFWriter(output, 'llama')
    target = f'blk.{layer}.attn_norm.weight'
    expected = {}
    try:
        for name, field in reader.fields.items():
            if name.startswith('GGUF.') or name == 'general.architecture':
                continue
            subtype = field.types[1] if field.types[0] == GGUFValueType.ARRAY else None
            value = len(mapping) if name == 'llama.block_count' else field.contents()
            writer.add_key_value(name, value, field.types[0], subtype)
        for dest, origin in plan.items():
            tensor = tensors[origin]
            data = tensor.data
            if operation == 'norm' and origin == target:
                data = data * np.float32(strength)
            expected[dest] = hashlib.sha256(data).hexdigest()
            writer.add_tensor(dest, data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
    finally:
        writer.close()
    actual = GGUFReader(output, 'r')
    hashes = {t.name: hashlib.sha256(t.data).hexdigest() for t in actual.tensors}
    if hashes != expected or actual.fields['llama.block_count'].contents() != len(mapping):
        raise RuntimeError('Rewritten tensor verification failed')
    return dict(operation=operation, layer=layer, strength=strength, source_sha256=digest(source),
                output_sha256=digest(output), block_map=mapping, tensors=hashes)
