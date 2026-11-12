"""Frozen tiny-model before/after test evaluation; local files only."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import resource
import threading
import time

from baseline import ROOT, INPUT, REFERENCE, SYSTEM, check_memory, compare, memory_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', action='store_true', help='Evaluate the verified saved case 2 adapter')
    args = parser.parse_args()
    phase = 'after' if args.after else 'before'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(ROOT / 'models/.hf-cache')
    output = ROOT / f'results/02/local-2026-11-11/case2-{phase}.jsonl'
    metadata = ROOT / f'results/02/local-2026-11-11/case2-{phase}.metadata.json'
    with metadata.open('x') as dest, output.open('x') as rows:
        result = {'phase': 'trained_adapter' if args.after else 'untrained', 'seed': 42, 'temperature': 0,
                  'max_tokens': 128, 'context_limit': 2048, 'system_prompt': SYSTEM,
                  'completed_indexes': [], 'memory_samples': []}
        start = time.perf_counter()
        stop, critical = threading.Event(), threading.Event()

        def sample(phase):
            state = memory_state()
            result['memory_samples'].append({'elapsed_seconds': time.perf_counter() - start,
                                             'phase': phase, **state})
            if state['pressure_level'] == 4 or state['free_percent'] is None or state['free_percent'] < 8:
                critical.set()
            return state

        def monitor():
            while not stop.wait(0.5):
                try:
                    sample('periodic')
                except Exception as exc:
                    result['monitor_error'] = str(exc)
                    critical.set()
                    return

        def guard(phase):
            check_memory(sample(phase))
            if critical.is_set():
                raise RuntimeError('Memory monitor requested stop')

        watcher = threading.Thread(target=monitor, daemon=True)
        model = stream = response = None
        try:
            guard('before_import')
            watcher.start()
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
            result['versions'] = {p: importlib.metadata.version(p) for p in
                                  ('mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'tokenizers')}
            provenance = json.loads((ROOT / 'results/02/local-2026-11-11/case2-model-provenance.json').read_text())
            model_path = ROOT / provenance['local_directory']
            for item in provenance['files']:
                with (model_path / item['name']).open('rb') as source:
                    assert hashlib.file_digest(source, 'sha256').hexdigest() == item['sha256']
            result['model_revision'] = provenance['revision']
            result['model_weight_sha256'] = provenance['files'][-1]['sha256']
            from freeze_split import build
            split = json.loads((ROOT / 'results/02/local-2026-11-11/case2-split.json').read_text())
            assert build() == split, 'Frozen split changed'
            result['split_sha256'] = split['split_sha256']
            result['dataset_sha256'] = split['dataset_sha256']
            data = json.loads((ROOT / 'data/dataset.json').read_text())
            adapter_path = None
            if args.after:
                training = json.loads((ROOT / 'results/02/local-2026-11-11/case2-train.json').read_text())
                assert training['status'] == 'completed'
                assert training['split_sha256'] == split['split_sha256']
                assert training['model_revision'] == provenance['revision']
                for item in training['adapter_files']:
                    artifact = ROOT / item['path']
                    assert artifact.stat().st_size == item['bytes']
                    with artifact.open('rb') as source:
                        assert hashlib.file_digest(source, 'sha256').hexdigest() == item['sha256']
                result['adapter_files'] = training['adapter_files']
                adapter_path = str(ROOT / 'models/case2-lora')
            guard('before_load')
            begin = time.perf_counter()
            model, tokenizer = load(str(model_path), adapter_path=adapter_path)
            mx.eval(model.parameters())
            model.eval()
            result['load_seconds'] = time.perf_counter() - begin
            mx.random.seed(42)
            for index in split['indexes']['test']:
                guard(f'before_input_{index}')
                row = data[index]
                messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row[INPUT]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                assert len(prompt) + result['max_tokens'] <= result['context_limit'], 'Context overflow'
                begin = time.perf_counter()
                pieces, tokens = [], []
                stream = stream_generate(model, tokenizer, prompt=prompt, max_tokens=result['max_tokens'],
                                         sampler=make_sampler(temp=0), prefill_step_size=128)
                for response in stream:
                    if critical.is_set():
                        raise RuntimeError('Memory monitor requested stop')
                    pieces.append(response.text)
                    tokens.append(response.token)
                text = ''.join(pieces)
                record = {'dataset_index': index, 'input': row[INPUT], 'reference': row[REFERENCE],
                          'prompt_token_ids': prompt, 'output': text, 'output_token_ids': tokens,
                          'elapsed_seconds': time.perf_counter() - begin,
                          'prompt_tokens': response.prompt_tokens, 'generation_tokens': response.generation_tokens,
                          'prompt_tps': response.prompt_tps, 'generation_tps': response.generation_tps,
                          'finish_reason': response.finish_reason, 'metrics': compare(text, row[REFERENCE])}
                rows.write(json.dumps(record) + '\n')
                rows.flush()
                os.fsync(rows.fileno())
                result['completed_indexes'].append(index)
                sample(f'after_input_{index}')
                print(index, record['finish_reason'], record['elapsed_seconds'], flush=True)
                stream = response = None
            result['mlx_peak_memory_bytes'] = mx.get_peak_memory()
            result['status'] = 'completed'
        except BaseException as exc:
            result['status'] = 'failed'
            result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if stream is not None:
                stream.close()
            model = stream = response = None
            gc.collect()
            if 'mx' in locals():
                mx.clear_cache()
            stop.set()
            if watcher.ident is not None:
                watcher.join(timeout=5)
            sample('after_release')
            result['total_seconds'] = time.perf_counter() - start
            result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            json.dump(result, dest, indent=2)
            dest.write('\n')


if __name__ == '__main__':
    main()
