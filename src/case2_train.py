"""Bounded local LoRA training on frozen train rows only; no held-out use."""
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
    path = ROOT / 'results/case2-train.json'
    with path.open('x') as dest:
        result = {'purpose': '20 LoRA optimizer updates; partial answers only; no held-out inspection',
                  'seed': 42, 'sequence_cap': 128, 'batch_size': 1, 'steps_requested': 20, 'learning_rate': 0.0001, 'memory_samples': []}
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
        model = grads = logits = loss = optimizer = initial = weights = batch = mask = None
        try:
            guard('before_import')
            watcher.start()
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            from mlx.utils import tree_flatten
            from mlx_lm import load
            from mlx_lm.tuner.utils import linear_to_lora_layers

            result['versions'] = {p: importlib.metadata.version(p) for p in
                                  ('mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'tokenizers')}
            provenance = json.loads((ROOT / 'results/case2-model-provenance.json').read_text())
            model_path = ROOT / 'models/SmolLM2-135M-Instruct'
            for item in provenance['files']:
                source = model_path / item['name']
                with source.open('rb') as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
            raw = (ROOT / 'data/dataset.json').read_bytes()
            split = json.loads((ROOT / 'results/case2-split.json').read_text())
            assert hashlib.sha256(raw).hexdigest() == split['dataset_sha256']
            from freeze_split import build
            assert build() == split
            result['dataset_sha256'] = split['dataset_sha256']
            result['model_revision'] = provenance['revision']
            result['model_files'] = provenance['files']
            result['split_sha256'] = split['split_sha256']
            data = json.loads(raw)
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
            model.train()
            optimizer = optim.Adam(learning_rate=result['learning_rate'], bias_correction=False)
            result['optimizer'] = {'name': 'mlx.optimizers.Adam', 'betas': [0.9, 0.999], 'eps': 1e-8,
                                   'weight_decay': 0, 'gradient_clipping': None, 'bias_correction': False}
            initial = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
            mx.eval(initial)
            result['steps'] = []
            result['skipped_indexes'] = []
            training_start = time.perf_counter()
            for index in split['indexes']['train']:
                row = data[index]
                messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row[INPUT]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': row[REFERENCE]}], tokenize=True)
                assert full[:len(prompt)] == prompt
                tokens = full[:result['sequence_cap'] + 1]
                if len(prompt) >= len(tokens) - 1:
                    result['skipped_indexes'].append(index)
                    continue
                batch = mx.array([tokens])
                mask = mx.arange(len(tokens) - 1)[None, :] >= len(prompt) - 1

                def objective(m):
                    scores = m(batch[:, :-1])
                    return (nn.losses.cross_entropy(scores, batch[:, 1:]) * mask).sum() / mask.sum()

                guard(f'before_step_{len(result["steps"]) + 1}')
                begin = time.perf_counter()
                loss, grads = nn.value_and_grad(model, objective)(model)
                mx.eval(loss, grads)
                finite = all(mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(grads))
                assert finite and mx.isfinite(loss).item(), 'Nonfinite loss or gradients'
                if critical.is_set():
                    raise RuntimeError('Memory monitor requested stop before optimizer update')
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                sample('after_optimizer_update')
                if critical.is_set():
                    raise RuntimeError('Memory monitor requested stop')
                record = {'step': len(result['steps']) + 1, 'dataset_index': index,
                          'prompt_tokens': len(prompt), 'sequence_tokens': len(tokens),
                          'supervised_tokens': len(tokens) - len(prompt), 'token_ids': tokens,
                          'loss_before_update': loss.item(), 'gradients_finite': finite,
                          'elapsed_seconds': time.perf_counter() - begin}
                result['steps'].append(record)
                print(record['step'], index, record['loss_before_update'], flush=True)
                if len(result['steps']) == result['steps_requested']:
                    break
            assert len(result['steps']) == result['steps_requested']
            result['training_seconds'] = time.perf_counter() - training_start
            weights = dict(tree_flatten(model.trainable_parameters()))
            result['adapter_delta_l2'] = sum(mx.sum((v.astype(mx.float32) - initial[k].astype(mx.float32)) ** 2).item()
                                             for k, v in weights.items()) ** 0.5
            assert result['adapter_delta_l2'] > 0
            adapter_dir = ROOT / 'models/case2-lora'
            adapter_dir.mkdir(exist_ok=False)
            adapter = adapter_dir / 'adapters.safetensors'
            mx.save_safetensors(str(adapter), weights)
            config_path = adapter_dir / 'adapter_config.json'
            config_path.write_text(json.dumps({'fine_tune_type': 'lora', 'num_layers': 2,
                                               'lora_parameters': config}, indent=2) + '\n')
            result['adapter_files'] = []
            for artifact in (adapter, config_path):
                with artifact.open('rb') as source:
                    digest = hashlib.file_digest(source, 'sha256').hexdigest()
                result['adapter_files'].append({'path': str(artifact.relative_to(ROOT)),
                                                'bytes': artifact.stat().st_size, 'sha256': digest})
            result['mlx_peak_memory_bytes'] = mx.get_peak_memory()
            sample('after_adapter_save')
            if critical.is_set():
                raise RuntimeError('Memory monitor requested stop')
            result['status'] = 'completed'
        except BaseException as exc:
            result['status'] = 'failed'
            result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            model = grads = logits = loss = optimizer = initial = weights = batch = mask = None
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
