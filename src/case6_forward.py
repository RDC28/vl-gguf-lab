"""Independent NumPy forward pass over unchanged decoded F16 GGUF weights."""
import argparse
import gc
import importlib.metadata
import json
import resource
from pathlib import Path
import threading
import time

import numpy as np
from gguf import GGUFReader

from baseline import ROOT, check_memory, memory_state
from case3_acquire import NAME, SHA256, digest


def forward(reader, tokens, config, guard, *, f16_inputs=False):
    tensors = {t.name: t for t in reader.tensors}

    def weight(name):
        tensor = tensors[name + '.weight']
        assert tensor.tensor_type.name in ('F16', 'F32')
        # GGUFReader exposes C-order arrays with the GGML dimensions reversed.
        return tensor.data.astype(np.float32)

    def linear(x, name):
        # F16 GGML CPU dot products round the other operand to F16 too.
        if f16_inputs:
            x = x.astype(np.float16).astype(np.float32)
        return x @ weight(name).T

    eps = config['llama.attention.layer_norm_rms_epsilon']
    heads = config['llama.attention.head_count']
    kv_heads = config['llama.attention.head_count_kv']
    width = config['llama.embedding_length']
    dim = width // heads
    length = len(tokens)
    angles = np.arange(length, dtype=np.float32)[:, None] * (
        np.float32(config['llama.rope.freq_base']) **
        (-np.arange(0, dim, 2, dtype=np.float32) / dim))[None, :]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]

    def norm(x, w):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps) * w

    def rope(x):
        # Llama GGUF uses adjacent rotary pairs, already permuted on conversion.
        a, b = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = a*cos - b*sin
        out[..., 1::2] = a*sin + b*cos
        return out

    embedding = weight('token_embd')
    x = embedding[tokens]
    for layer in range(config['llama.block_count']):
        guard('before_block:' + str(layer))
        prefix = f'blk.{layer}.'
        z = norm(x, weight(prefix + 'attn_norm'))
        q = rope((linear(z, prefix + 'attn_q')).reshape(length, heads, dim))
        k = rope((linear(z, prefix + 'attn_k')).reshape(length, kv_heads, dim))
        v = (linear(z, prefix + 'attn_v')).reshape(length, kv_heads, dim)
        k = np.repeat(k, heads // kv_heads, axis=1)
        v = np.repeat(v, heads // kv_heads, axis=1)
        scores = np.einsum('thd,shd->hts', q, k) / np.sqrt(np.float32(dim))
        scores = np.where(np.triu(np.ones((length, length), dtype=bool), 1), -np.inf, scores)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        attention = np.einsum('hts,shd->thd', probs, v).reshape(length, width)
        x = x + linear(attention, prefix + 'attn_output')
        z = norm(x, weight(prefix + 'ffn_norm'))
        gate = linear(z, prefix + 'ffn_gate')
        up = linear(z, prefix + 'ffn_up')
        x = x + linear((gate / (1 + np.exp(-gate))) * up, prefix + 'ffn_down')
    return linear(norm(x, weight('output_norm')), 'token_embd')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--f16-inputs', action='store_true',
                        help='Round linear operands to F16 before float32 accumulation')
    args = parser.parse_args()
    stem = 'case6-forward-f16-inputs' if args.f16_inputs else 'case6-forward'
    path = ROOT / f'results/06/local-2026-11-11/{stem}.json'
    output = ROOT / f'results/06/local-2026-11-11/{stem}-logits.npz'
    if path.exists() or output.exists():
        raise FileExistsError('Forward evidence already exists')
    start = time.perf_counter()
    reference_path = ROOT / 'results/06/local-2026-11-11/case6-reference.json'
    reference = json.loads(reference_path.read_text())
    source = ROOT / 'models' / NAME
    result = dict(source_sha256=SHA256, reference_sha256=digest(reference_path),
                  numpy_version=importlib.metadata.version('numpy'),
                  f16_linear_inputs=args.f16_inputs,
                  implementation_sha256=digest(Path(__file__)),
                  cpu_source_sha256=digest(ROOT / 'results/06/local-2026-11-11/case6-cpu-source.c'),
                  implementation='NumPy float32, full 30-block decoded F16 source; causal full-sequence attention; no KV cache',
                  memory_samples=[], acceptance=reference['acceptance'])
    stop, critical = threading.Event(), threading.Event()
    reader = logits = None

    def sample(phase):
        state = memory_state()
        result['memory_samples'].append(dict(phase=phase, elapsed_seconds=time.perf_counter()-start, **state))
        if state['pressure_level'] == 4 or state['free_percent'] is None or state['free_percent'] < 8:
            critical.set()
        return state

    def guard(phase):
        sample(phase)
        if critical.is_set():
            raise RuntimeError('Memory monitor requested stop')

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
            assert digest(source) == SHA256
            saved_path = ROOT / 'results/06/local-2026-11-11/case6-reference-logits.npz'
            assert digest(saved_path) == reference['logits_sha256']
            with np.load(saved_path) as saved:
                expected = saved['logits'].copy()
                tokens = saved['token_ids'].tolist()
            assert tokens == reference['token_ids']
            reader = GGUFReader(source, 'r')
            config = reference['architecture']
            for key, value in config.items():
                assert reader.fields[key].contents() == value
            result['token_ids'] = tokens
            check_memory(sample('before_input'))
            begin = time.perf_counter()
            logits = forward(reader, tokens, config, guard, f16_inputs=args.f16_inputs)
            result['forward_seconds'] = time.perf_counter()-begin
            guard('after_forward')
            assert logits.shape == expected.shape and np.isfinite(logits).all()
            error = np.abs(logits.astype(np.float64)-expected)
            tolerance = result['acceptance']['atol'] + result['acceptance']['rtol']*np.abs(expected)
            result.update(max_absolute_error=float(error.max()), rms_error=float(np.sqrt(np.mean(error**2))),
                          violating_logits=int((error > tolerance).sum()), total_logits=int(error.size),
                          criterion_passed=bool(np.all(error <= tolerance)),
                          per_position_max_error=error.max(axis=-1).tolist(),
                          argmax_ids=logits.argmax(axis=-1).tolist(),
                          argmax_agreement=(logits.argmax(axis=-1) == expected.argmax(axis=-1)).tolist())
            with output.open('xb') as stream:
                np.savez_compressed(stream, logits=logits, token_ids=np.array(tokens, dtype=np.int32))
            result.update(logits_sha256=digest(output), status='comparison_completed')
            assert digest(source) == SHA256
        except BaseException as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            reader = logits = None
            gc.collect()
            stop.set()
            if watcher.ident is not None:
                watcher.join(timeout=5)
            sample('after_release')
            result.update(total_seconds=time.perf_counter()-start,
                          peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          limitations='Five fixed positions only. No tokenizer reimplementation, generation, training or vision. F16 weights decoded exactly to F32; no packed quantization. Full-sequence attention differs from sequential reference evaluation. RSS excludes some unified-memory use; sampling can miss transients.')
            json.dump(result, dest, indent=2)
            dest.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'memory_samples'}, indent=2))


if __name__ == '__main__':
    main()
