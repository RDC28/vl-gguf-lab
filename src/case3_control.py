"""Sequential source/rewrite inference smoke control, with guarded streaming."""
import hashlib
import argparse
import importlib.metadata
import json
import resource
import threading
import time

from baseline import ROOT, INPUT, SYSTEM, check_memory, memory_state
from case3_acquire import NAME, SHA256, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--pruned', action='store_true', help='Evaluate the verified removed block against saved source output')
    mode.add_argument('--expanded', action='store_true', help='Evaluate the verified added block against saved source output')
    mode.add_argument('--edited', action='store_true', help='Evaluate the verified norm edit against saved source output')
    args = parser.parse_args()
    variant = args.edited or args.expanded or args.pruned
    path = ROOT / ('results/case5-pruned.json' if args.pruned else 'results/case4-expanded.json' if args.expanded else 'results/case3-edited.json' if args.edited else 'results/case3-control.json')
    start = time.perf_counter()
    raw = (ROOT / 'data/dataset.json').read_bytes()
    dataset_hash = hashlib.sha256(raw).hexdigest()
    assert dataset_hash == '073bd194bdb79e3ecfcfad51e09f23510ca697431475221b785c369619d7b8cf'
    scene = json.loads(raw)[6][INPUT]
    settings = dict(n_ctx=2048, n_gpu_layers=0, n_batch=128, n_threads=4,
                    n_threads_batch=4, seed=42, use_mmap=True)
    decoding = dict(temperature=0, top_p=1, top_k=1, repeat_penalty=1, max_tokens=128)
    result = dict(dataset_sha256=dataset_hash, dataset_index=6, input=scene,
                  system_prompt=SYSTEM, settings=settings, decoding=decoding, runs=[])
    artifacts = [(NAME, SHA256), ('SmolLM2-135M-Instruct-control.gguf', SHA256)]
    if variant:
        control_path = ROOT / 'results/case3-control.json'
        control = json.loads(control_path.read_text())
        assert control['status'] == 'completed' and control['outputs_identical']
        for key in ('dataset_sha256', 'dataset_index', 'input', 'system_prompt', 'settings', 'decoding'):
            assert result[key] == control[key], f'Control mismatch: {key}'
        edit_path = ROOT / ('results/case5-remove.json' if args.pruned else 'results/case4-expand.json' if args.expanded else 'results/case3-perturb.json')
        edit = json.loads(edit_path.read_text())
        assert edit['status'] == 'completed' and edit['source_sha256'] == SHA256
        artifacts = [(str((ROOT / edit['output_path']).relative_to(ROOT / 'models')), edit['output_sha256'])]
        result.update(control_evidence_sha256=digest(control_path), edit_evidence_sha256=digest(edit_path))
    with path.open('x') as dest:
        try:
            check_memory()
            from llama_cpp import Llama, llama_cpp
            result['llama_cpp_python'] = importlib.metadata.version('llama-cpp-python')
            result['system_info'] = llama_cpp.llama_print_system_info().decode()
            if variant:
                assert result['llama_cpp_python'] == control['llama_cpp_python']
                assert result['system_info'] == control['system_info']
            for name, expected_hash in artifacts:
                source = ROOT / 'models' / name
                assert digest(source) == expected_hash
                run = dict(model=name, sha256=expected_hash, memory_samples=[])
                result['runs'].append(run)
                stop, critical = threading.Event(), threading.Event()
                model = None

                def sample(phase):
                    state = memory_state()
                    run['memory_samples'].append(dict(phase=phase, elapsed_seconds=time.perf_counter() - start, **state))
                    if state['pressure_level'] == 4 or state['free_percent'] is None or state['free_percent'] < 8:
                        critical.set()
                    return state

                def monitor():
                    while not stop.wait(0.5):
                        try:
                            sample('periodic')
                        except Exception as exc:
                            run['monitor_error'] = str(exc)
                            critical.set()
                            return

                watcher = threading.Thread(target=monitor, daemon=True)
                try:
                    check_memory(sample('before_load'))
                    watcher.start()
                    begin = time.perf_counter()
                    model = Llama(model_path=str(source), no_perf=False, verbose=True, **settings)
                    run['load_seconds'] = time.perf_counter() - begin
                    run['chat_format'] = model.chat_format
                    run['embedded_chat_template'] = model.metadata.get('tokenizer.chat_template')
                    if variant:
                        for key in ('chat_format', 'embedded_chat_template'):
                            assert run[key] == control['runs'][0][key], f'Control mismatch: {key}'
                    check_memory(sample('before_input'))
                    if critical.is_set():
                        raise RuntimeError('Memory monitor requested stop')
                    llama_cpp.llama_perf_context_reset(model.ctx)
                    begin = time.perf_counter()
                    chunks = model.create_chat_completion(messages=[{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': scene}], stream=True, **decoding)
                    parts, finish = [], None
                    try:
                        for chunk in chunks:
                            if critical.is_set():
                                raise RuntimeError('Memory monitor requested stop')
                            choice = chunk['choices'][0]
                            parts.append(choice['delta'].get('content') or '')
                            finish = choice.get('finish_reason') or finish
                    finally:
                        chunks.close()
                    timings = llama_cpp.llama_perf_context(model.ctx)
                    run.update(wall_seconds=time.perf_counter() - begin, output=''.join(parts),
                               finish_reason=finish, prompt_eval_tokens=int(timings.n_p_eval),
                               completion_eval_tokens=int(timings.n_eval),
                               prompt_eval_seconds=float(timings.t_p_eval_ms) / 1000,
                               generation_eval_seconds=float(timings.t_eval_ms) / 1000)
                    sample('after_input')
                finally:
                    if model is not None:
                        model.close()
                    stop.set()
                    if watcher.ident is not None:
                        watcher.join(timeout=5)
                    sample('after_close')
                    run['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                assert digest(source) == expected_hash
            if variant:
                from baseline import compare, REFERENCE
                reference = json.loads(raw)[6][REFERENCE]
                result['outputs_identical'] = result['runs'][0]['output'] == control['runs'][0]['output']
                result['original_metrics'] = compare(control['runs'][0]['output'], reference)
                result['edited_metrics'] = compare(result['runs'][0]['output'], reference)
                assert digest(ROOT / 'models' / NAME) == SHA256
            else:
                result['outputs_identical'] = result['runs'][0]['output'] == result['runs'][1]['output']
                assert result['outputs_identical'], 'Byte-identical control produced different text'
            result['status'] = 'completed'
        except BaseException as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            result['total_seconds'] = time.perf_counter() - start
            result['limitations'] = 'One scene, 128-token cap, CPU only. No logit parity or quality inference. RSS is cumulative process high-water mark and excludes some unified-memory allocations.'
            json.dump(result, dest, indent=2)
            dest.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
