"""No-change small GGUF rewrite control; verify all metadata and tensor bytes."""
import hashlib
import importlib.metadata
import json
import resource
import time

from baseline import ROOT, check_memory, memory_state
from case3_acquire import NAME, SHA256, digest


def inventory(reader):
    return {
        'fields': {name: {'types': [t.name for t in field.types],
                          'value_sha256': hashlib.sha256(json.dumps(field.contents(), ensure_ascii=False).encode()).hexdigest()}
                   for name, field in reader.fields.items()},
        'tensors': [{'name': t.name, 'shape': t.shape.tolist(), 'type': t.tensor_type.name,
                     'sha256': hashlib.sha256(t.data).hexdigest()} for t in reader.tensors],
    }


def main():
    output = ROOT / 'models/SmolLM2-135M-Instruct-control.gguf'
    result_path = ROOT / 'results/case3-rewrite.json'
    if output.exists() or result_path.exists():
        raise FileExistsError('Control or evidence already exists; refusing overwrite')
    start = time.perf_counter()
    result = {'purpose': 'No-change GGUFWriter control; no inference or perturbation',
              'source_sha256': SHA256, 'output_path': str(output.relative_to(ROOT)),
              'memory_samples': []}

    def sample(phase):
        state = memory_state()
        result['memory_samples'].append({'phase': phase, 'elapsed_seconds': time.perf_counter() - start, **state})
        check_memory(state)

    with result_path.open('x') as dest:
        writer = None
        try:
            sample('before_import')
            from gguf import GGUFReader, GGUFWriter, GGUFValueType
            result['gguf_version'] = importlib.metadata.version('gguf')
            source = ROOT / 'models' / NAME
            assert digest(source) == SHA256
            reader = GGUFReader(source, 'r')
            original = inventory(reader)
            # Match the independently saved acquisition inventory before writing.
            saved = json.loads((ROOT / 'results/case3-source.json').read_text())
            for name, field in original['fields'].items():
                assert field['types'] == saved['fields'][name]['types']
                assert field['value_sha256'] == saved['fields'][name]['value_sha256']
            for tensor, previous in zip(original['tensors'], saved['tensors'], strict=True):
                assert all(tensor[k] == previous[k] for k in tensor)
            writer = GGUFWriter(output, reader.fields['general.architecture'].contents())
            for name, field in reader.fields.items():
                if name.startswith('GGUF.') or name == 'general.architecture':
                    continue
                subtype = field.types[1] if field.types[0] == GGUFValueType.ARRAY else None
                writer.add_key_value(name, field.contents(), field.types[0], subtype)
            for tensor in reader.tensors:
                sample('register:' + tensor.name)
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            sample('before_write')
            begin = time.perf_counter()
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=False)
            writer.close()
            writer = None
            result['write_seconds'] = time.perf_counter() - begin
            sample('after_write')
            control = GGUFReader(output, 'r')
            rewritten = inventory(control)
            assert original == rewritten, 'Metadata or tensor inventory changed'
            assert digest(source) == SHA256, 'Original source changed'
            result.update(status='completed', metadata_fields_verified=len(original['fields']),
                          tensors_verified=len(original['tensors']), output_sha256=digest(output),
                          output_size_bytes=output.stat().st_size,
                          byte_identical_file=digest(output) == SHA256,
                          metadata_and_tensor_bytes_identical=True)
            sample('after_verification')
        except BaseException as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if writer is not None:
                writer.close()
            result['elapsed_seconds'] = time.perf_counter() - start
            result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['limitations'] = 'No inference or load equivalence tested. Phase/tensor memory snapshots, not continuous monitoring. RSS excludes some unified-memory allocations.'
            json.dump(result, dest, indent=2)
            dest.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'memory_samples'}, indent=2))


if __name__ == '__main__':
    main()
