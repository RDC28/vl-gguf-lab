"""Capture unchanged tiny GGUF logits before implementing a custom forward pass."""
import importlib.metadata
import json
import resource
import threading
import time

import numpy as np

from baseline import ROOT, check_memory, memory_state
from case3_acquire import NAME, SHA256, digest


def main():
    path = ROOT / 'results/case6-reference.json'
    logits_path = ROOT / 'results/case6-reference-logits.npz'
    if path.exists() or logits_path.exists():
        raise FileExistsError('Reference evidence already exists')
    start = time.perf_counter()
    source = ROOT / 'models' / NAME
    assert digest(source) == SHA256
    inventory = json.loads((ROOT / 'results/case3-source.json').read_text())
    result = dict(source_sha256=SHA256, source_model=NAME,
                  inventory_sha256=digest(ROOT / 'results/case3-source.json'),
                  architecture={k: v['value'] for k, v in inventory['fields'].items()
                                if k.startswith('llama.') or k == 'general.architecture'},
                  text='The camera moves slowly.', add_bos=False, special=False,
                  memory_samples=[],
                  acceptance=dict(atol=0.001, rtol=0.001,
                                  rule='Every saved logit must satisfy abs(custom-reference) <= atol + rtol*abs(reference); report max/RMS errors and argmax agreement separately.'))
    stop, critical = threading.Event(), threading.Event()
    model = None

    def sample(phase):
        state = memory_state()
        result['memory_samples'].append(dict(phase=phase, elapsed_seconds=time.perf_counter()-start, **state))
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

    watcher = threading.Thread(target=monitor, daemon=True)
    with path.open('x') as dest:
        try:
            check_memory(sample('before_load'))
            watcher.start()
            from llama_cpp import Llama, llama_cpp
            result['llama_cpp_python'] = importlib.metadata.version('llama-cpp-python')
            result['system_info'] = llama_cpp.llama_print_system_info().decode()
            settings = dict(n_ctx=64, n_batch=16, n_ubatch=16, n_gpu_layers=0,
                            offload_kqv=False, op_offload=False, flash_attn=False,
                            n_threads=4, n_threads_batch=4, seed=42, use_mmap=True,
                            logits_all=True, type_k=llama_cpp.GGML_TYPE_F32,
                            type_v=llama_cpp.GGML_TYPE_F32)
            result['settings'] = settings
            begin = time.perf_counter()
            model = Llama(model_path=str(source), verbose=True, **settings)
            result['load_seconds'] = time.perf_counter()-begin
            tokens = model.tokenize(result['text'].encode(), add_bos=False, special=False)
            assert 1 < len(tokens) <= 16
            result['token_ids'] = tokens
            result['token_pieces_hex'] = [model.detokenize([t]).hex() for t in tokens]
            check_memory(sample('before_input'))
            begin = time.perf_counter()
            # One call per position permits safety checks and exercises the real KV cache.
            for token in tokens:
                if critical.is_set():
                    raise RuntimeError('Memory monitor requested stop')
                model.eval([token])
                sample('after_token')
            result['eval_seconds'] = time.perf_counter()-begin
            if critical.is_set():
                raise RuntimeError('Memory monitor requested stop')
            logits = model.scores[:len(tokens)].copy()
            assert logits.shape == (len(tokens), model.n_vocab()) and np.isfinite(logits).all()
            with logits_path.open('xb') as output:
                np.savez_compressed(output, logits=logits, token_ids=np.array(tokens, dtype=np.int32))
            result.update(logits_shape=list(logits.shape), logits_dtype=str(logits.dtype),
                          logits_sha256=digest(logits_path), argmax_ids=logits.argmax(axis=-1).tolist(),
                          status='reference_saved')
        except BaseException as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if model is not None:
                model.close()
            stop.set()
            if watcher.ident is not None:
                watcher.join(timeout=5)
            sample('after_close')
            result['total_seconds'] = time.perf_counter()-start
            result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['limitations'] = 'Reference capture only, no custom forward/parity or training. No chat template, dataset or references used. Float32 KV cache selected to simplify later parity. RSS excludes some unified-memory allocations; samples can miss transients.'
            json.dump(result, dest, indent=2)
            dest.write('\n')
    assert digest(source) == SHA256
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
