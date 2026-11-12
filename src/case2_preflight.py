"""One local forward/backward feasibility check; no optimizer update or held-out use."""
import gc
import hashlib
import importlib.metadata
import json
import os
import resource
import threading
import time

from baseline import ROOT, INPUT, REFERENCE, SYSTEM, check_memory, memory_state


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(ROOT / 'models/.hf-cache')
    path = ROOT / 'results/02/local-2026-11-11/case2-preflight.json'
    with path.open('x') as dest:
        result = {'purpose': 'Feasibility only; no optimizer step, no saved adapter, no held-out inspection',
                  'seed': 42, 'sequence_cap': 128, 'batch_size': 1, 'memory_samples': []}
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
        model = grads = logits = loss = None
        try:
            guard('before_import')
            watcher.start()
            import mlx.core as mx
            import mlx.nn as nn
            from mlx.utils import tree_flatten
            from mlx_lm import load
            from mlx_lm.tuner.utils import linear_to_lora_layers

            result['versions'] = {p: importlib.metadata.version(p) for p in
                                  ('mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'tokenizers')}
            provenance = json.loads((ROOT / 'results/02/local-2026-11-11/case2-model-provenance.json').read_text())
            model_path = ROOT / 'models/SmolLM2-135M-Instruct'
            for item in provenance['files']:
                source = model_path / item['name']
                with source.open('rb') as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
            raw = (ROOT / 'data/dataset.json').read_bytes()
            split = json.loads((ROOT / 'results/02/local-2026-11-11/case2-split.json').read_text())
            assert hashlib.sha256(raw).hexdigest() == split['dataset_sha256']
            result['split_sha256'] = split['split_sha256']
            index = split['indexes']['train'][0]
            result['dataset_index'] = index
            row = json.loads(raw)[index]
            guard('before_load')
            begin = time.perf_counter()
            model, tokenizer = load(str(model_path))
            mx.eval(model.parameters())
            result['load_seconds'] = time.perf_counter() - begin
            mx.random.seed(42)
            model.freeze()
            config = {'rank': 4, 'scale': 8.0, 'dropout': 0.0,
                      'keys': ['self_attn.q_proj', 'self_attn.v_proj']}
            linear_to_lora_layers(model, 2, config)
            result['lora'] = {'last_layers': 2, **config}
            result['trainable_parameters'] = {k: list(v.shape) for k, v in tree_flatten(model.trainable_parameters())}
            result['trainable_elements'] = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row[INPUT]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': row[REFERENCE]}], tokenize=True)
            assert full[:len(prompt)] == prompt
            tokens = full[:129]
            assert len(prompt) < len(tokens) - 1, 'No supervised tokens in short sequence'
            result.update(prompt_tokens=len(prompt), sequence_tokens=len(tokens),
                          supervised_tokens=len(tokens) - len(prompt), token_ids=tokens)
            batch = mx.array([tokens])
            mask = mx.arange(len(tokens) - 1)[None, :] >= len(prompt) - 1

            def objective(m):
                scores = m(batch[:, :-1])
                return (nn.losses.cross_entropy(scores, batch[:, 1:]) * mask).sum() / mask.sum()

            guard('before_forward_backward')
            begin = time.perf_counter()
            loss, grads = nn.value_and_grad(model, objective)(model)
            mx.eval(loss, grads)
            result['forward_backward_seconds'] = time.perf_counter() - begin
            result['loss'] = loss.item()
            result['gradients_finite'] = all(mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(grads))
            result['gradient_l2'] = sum(mx.sum(v.astype(mx.float32) ** 2).item() for _, v in tree_flatten(grads)) ** 0.5
            result['mlx_peak_memory_bytes'] = mx.get_peak_memory()
            sample('after_forward_backward')
            if critical.is_set():
                raise RuntimeError('Memory monitor requested stop')
            assert result['gradients_finite'] and result['gradient_l2'] > 0
            result['status'] = 'completed'
        except BaseException as exc:
            result['status'] = 'failed'
            result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            model = grads = logits = loss = None
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
        print(json.dumps({k: v for k, v in result.items() if k != 'memory_samples'}, indent=2))


if __name__ == '__main__':
    main()
