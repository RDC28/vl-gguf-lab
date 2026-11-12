"""Remove predetermined interior block 15 from the tiny F16 GGUF; verify the map."""
import json
import resource
import time

from baseline import ROOT, check_memory, memory_state
from case3_acquire import NAME, SHA256, digest
from case3_rewrite import inventory

COUNT_KEY = 'llama.block_count'


def main():
    output = ROOT / 'models/SmolLM2-135M-Instruct-minus1.gguf'
    evidence = ROOT / 'results/05/local-2026-11-11/case5-remove.json'
    if output.exists() or evidence.exists():
        raise FileExistsError('Edited artifact or evidence exists; refusing overwrite')
    start = time.perf_counter()
    result = dict(source_sha256=SHA256, output_path=str(output.relative_to(ROOT)),
                  operation='remove original block 15; shift original blocks 16-29 down one; no training',
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
            source = ROOT / 'models' / NAME
            assert digest(source) == SHA256
            reader = GGUFReader(source, 'r')
            original = inventory(reader)
            saved = json.loads((ROOT / 'results/03/local-2026-11-11/case3-source.json').read_text())
            assert original['fields'] == {k: {x: v[x] for x in ('types', 'value_sha256')} for k, v in saved['fields'].items()}
            for tensor, previous in zip(original['tensors'], saved['tensors'], strict=True):
                assert all(tensor[k] == previous[k] for k in tensor)
            assert reader.fields['general.architecture'].contents() == 'llama'
            assert reader.fields[COUNT_KEY].contents() == 30
            block = [t for t in reader.tensors if t.name.startswith('blk.15.')]
            assert len(block) == 9
            mapping = {}
            for tensor in reader.tensors:
                parts = tensor.name.split('.')
                if parts[0] == 'blk':
                    index = int(parts[1])
                    if index == 15:
                        continue
                    parts[1] = str(index - (index > 15))
                mapping['.'.join(parts)] = tensor.name
            result.update(original_blocks=30, pruned_blocks=29, removed_block=15,
                          selection_rule='predetermined interior block; no calibration or score selection',
                          new_to_source_block_map={str(i): i + (i >= 15) for i in range(29)},
                          retained_tensor_map=mapping,
                          removed_elements=sum(int(t.n_elements) for t in block),
                          removed_tensor_bytes=sum(int(t.n_bytes) for t in block),
                          removed_tensor_inventory=[t for t in original['tensors'] if t['name'].startswith('blk.15.')])
            writer = GGUFWriter(output, reader.fields['general.architecture'].contents())
            for name, field in reader.fields.items():
                if name.startswith('GGUF.') or name == 'general.architecture':
                    continue
                subtype = field.types[1] if field.types[0] == GGUFValueType.ARRAY else None
                writer.add_key_value(name, 29 if name == COUNT_KEY else field.contents(), field.types[0], subtype)
            source_tensors = {t.name: t for t in reader.tensors}
            for new_name, source_name in mapping.items():
                sample('register:' + new_name)
                tensor = source_tensors[source_name]
                writer.add_tensor(new_name, tensor.data, raw_dtype=tensor.tensor_type)
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
            changed_fields = [k for k in original['fields'] if original['fields'][k] != after['fields'][k]]
            assert set(original['fields']) == set(after['fields'])
            assert set(changed_fields) == {'GGUF.tensor_count', COUNT_KEY}, changed_fields
            assert changed_reader.fields[COUNT_KEY].contents() == 29
            assert changed_reader.fields['GGUF.tensor_count'].contents() == 263
            assert changed_reader.fields[COUNT_KEY].types == reader.fields[COUNT_KEY].types
            originals = {t['name']: t for t in original['tensors']}
            assert len(after['tensors']) == 263
            assert {t['name'] for t in after['tensors']} == set(mapping)
            for tensor in after['tensors']:
                expected = dict(originals[mapping[tensor['name']]], name=tensor['name'])
                assert tensor == expected
            assert digest(source) == SHA256
            result.update(status='completed', byte_preserved_retained_tensors=263,
                          changed_metadata_fields=changed_fields,
                          unchanged_metadata_fields=len(original['fields']) - 2,
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
            result['limitations'] = 'Artifact verification only; pruned model not loaded or evaluated. Fixed block removal has no influence calibration; no behavior preservation or quality gain assumed. Phase/tensor snapshots are not continuous monitoring. RSS excludes some unified-memory allocations.'
            json.dump(result, dest, indent=2)
            dest.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'memory_samples'}, indent=2))


if __name__ == '__main__':
    main()
