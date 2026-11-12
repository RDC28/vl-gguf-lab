"""Apply the predeclared F32 norm edit, preserving all unrelated GGUF data."""
import json
import resource
import time

import numpy as np

from baseline import ROOT, check_memory, memory_state
from case3_acquire import NAME, SHA256, digest
from case3_rewrite import inventory

TARGET = 'blk.0.attn_norm.weight'


def main():
    output = ROOT / 'models/SmolLM2-135M-Instruct-norm101.gguf'
    evidence = ROOT / 'results/03/local-2026-11-11/case3-perturb.json'
    if output.exists() or evidence.exists():
        raise FileExistsError('Edited artifact or evidence exists; refusing overwrite')
    start = time.perf_counter()
    result = dict(source_sha256=SHA256, output_path=str(output.relative_to(ROOT)),
                  target=TARGET, operation='multiply by numpy.float32(1.01)',
                  memory_samples=[])

    def sample(phase):
        state = memory_state()
        result['memory_samples'].append(dict(phase=phase, elapsed_seconds=time.perf_counter() - start, **state))
        check_memory(state)

    with evidence.open('x') as dest:
        writer = None
        try:
            sample('before_import')
            from gguf import GGUFReader, GGUFWriter, GGUFValueType
            import importlib.metadata
            result['gguf_version'] = importlib.metadata.version('gguf')
            result['numpy_version'] = np.__version__
            source = ROOT / 'models' / NAME
            assert digest(source) == SHA256
            reader = GGUFReader(source, 'r')
            original = inventory(reader)
            saved = json.loads((ROOT / 'results/03/local-2026-11-11/case3-source.json').read_text())
            assert original['fields'] == {k: {x: v[x] for x in ('types', 'value_sha256')} for k, v in saved['fields'].items()}
            for tensor, previous in zip(original['tensors'], saved['tensors'], strict=True):
                assert all(tensor[k] == previous[k] for k in tensor)
            target = next(t for t in reader.tensors if t.name == TARGET)
            assert target.tensor_type.name == 'F32' and target.shape.tolist() == [576]
            edited = target.data.copy() * np.float32(1.01)
            assert edited.dtype == np.float32 and np.isfinite(edited).all()
            result.update(shape=target.shape.tolist(), dtype=target.tensor_type.name,
                          changed_elements=int(np.count_nonzero(edited != target.data)),
                          before_min=float(target.data.min()), before_max=float(target.data.max()),
                          after_min=float(edited.min()), after_max=float(edited.max()),
                          maximum_absolute_delta=float(np.abs(edited - target.data).max()))
            writer = GGUFWriter(output, reader.fields['general.architecture'].contents())
            for name, field in reader.fields.items():
                if name.startswith('GGUF.') or name == 'general.architecture':
                    continue
                subtype = field.types[1] if field.types[0] == GGUFValueType.ARRAY else None
                writer.add_key_value(name, field.contents(), field.types[0], subtype)
            for tensor in reader.tensors:
                sample('register:' + tensor.name)
                writer.add_tensor(tensor.name, edited if tensor.name == TARGET else tensor.data,
                                  raw_dtype=tensor.tensor_type)
            sample('before_write')
            begin = time.perf_counter()
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=False)
            writer.close()
            writer = None
            result['write_seconds'] = time.perf_counter() - begin
            sample('after_write')
            changed_reader = GGUFReader(output, 'r')
            after = inventory(changed_reader)
            assert original['fields'] == after['fields']
            changed = []
            for before, new in zip(original['tensors'], after['tensors'], strict=True):
                assert all(before[k] == new[k] for k in ('name', 'shape', 'type'))
                if before['sha256'] != new['sha256']:
                    changed.append(before['name'])
                    result['target_before_sha256'] = before['sha256']
                    result['target_after_sha256'] = new['sha256']
            assert changed == [TARGET], changed
            actual = next(t for t in changed_reader.tensors if t.name == TARGET)
            assert np.array_equal(actual.data, edited)
            assert digest(source) == SHA256
            result.update(status='completed', changed_tensors=changed,
                          unchanged_tensors=len(original['tensors']) - 1,
                          metadata_fields_verified=len(original['fields']),
                          output_sha256=digest(output), output_size_bytes=output.stat().st_size)
            sample('after_verification')
        except BaseException as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if writer is not None:
                writer.close()
            result['elapsed_seconds'] = time.perf_counter() - start
            result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['limitations'] = 'Artifact verification only; edited model not loaded or evaluated. Phase/tensor snapshots are not continuous monitoring. RSS excludes some unified-memory allocations.'
            json.dump(result, dest, indent=2)
            dest.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'memory_samples'}, indent=2))


if __name__ == '__main__':
    main()
