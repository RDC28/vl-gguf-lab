"""Text-only, bounded local baseline. Each output row is durable before the next input."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import re
import resource
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
INPUT = "user's scene description"
REFERENCE = 'shot breakdown'
SYSTEM = 'You are a film storyboard assistant. Turn the scene description into a detailed shot breakdown. Use numbered SHOT headings and describe the framing, camera movement, action, and sound for each shot.'


def memory_state():
    pressure = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
    report = subprocess.check_output(['memory_pressure'], text=True)
    match = re.search(r'System-wide memory free percentage: (\d+)%', report)
    return {'pressure_level': int(pressure), 'free_percent': int(match[1]) if match else None}


def check_memory():
    state = memory_state()
    # macOS pressure levels: normal=1, warning=2, critical=4.
    if state['pressure_level'] != 1 or state['free_percent'] is None or state['free_percent'] < 15:
        raise RuntimeError(f'Insufficient memory headroom; close other work before retrying: {state}')
    return state


def compare(output, reference):
    words = lambda s: re.findall(r'\b\w+\b', s.lower())
    a, b = Counter(words(output)), Counter(words(reference))
    common = sum((a & b).values())
    precision = common / sum(a.values()) if a else 0
    recall = common / sum(b.values()) if b else 0
    return {'output_words': sum(a.values()), 'reference_words': sum(b.values()),
            'output_shot_headings': len(re.findall(r'(?im)^\s*(?:\d+[.)]\s+|[-+*]\s+)?(?:\*{0,2}|#{1,6}\s*)SHOT\s+\d+', output)),
            'reference_shot_headings': len(re.findall(r'(?im)^\s*(?:\d+[.)]\s+|[-+*]\s+)?(?:\*{0,2}|#{1,6}\s*)SHOT\s+\d+', reference)),
            'unigram_precision': precision, 'unigram_recall': recall,
            'unigram_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0}


def run(dataset=ROOT / 'data/dataset.json', output=ROOT / 'results/baseline.jsonl', limit=10, seed=42, max_tokens=1024, n_ctx=2048):
    if not 1 <= limit <= 150 or not 1 <= max_tokens < n_ctx:
        raise ValueError('Require 1 <= limit <= 150 and 1 <= max_tokens < n_ctx')
    dataset, output = Path(dataset), Path(output)
    data_bytes = dataset.read_bytes()
    records = json.loads(data_bytes)
    if not isinstance(records, list) or not records or any(set(r) != {INPUT, REFERENCE} or any(not isinstance(v, str) for v in r.values()) for r in records):
        raise ValueError('Expected list of scene description / shot breakdown string pairs')
    if limit > len(records):
        raise ValueError('Sample exceeds dataset size')
    indexes = sorted(random.Random(seed).sample(range(len(records)), limit))
    output.parent.mkdir(exist_ok=True, parents=True)
    # Exclusive create avoids silently overwriting raw experimental evidence.
    with (ROOT / 'models/.inference.lock').open('a') as lock, output.open('x') as dest:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((ROOT / 'models/manifest.json').read_text())
        model_path = ROOT / 'models' / manifest['files'][0]['name']
        before = check_memory()
        backend_started = time.perf_counter()
        from llama_cpp import Llama, llama_cpp
        system_info = llama_cpp.llama_print_system_info().decode()
        backend_init_seconds = time.perf_counter() - backend_started
        metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'dataset_sha256': hashlib.sha256(data_bytes).hexdigest(),
                    'dataset_rows': len(records), 'sample_indexes': indexes, 'model': manifest,
                    'python': platform.python_version(), 'macos': platform.mac_ver()[0], 'machine': platform.machine(),
                    'llama_cpp_python': importlib.metadata.version('llama-cpp-python'),
                    'llama_system_info': system_info, 'backend_init_seconds': backend_init_seconds,
                    'settings': {'seed': seed, 'temperature': 0, 'top_p': 1, 'top_k': 1, 'repeat_penalty': 1,
                                 'n_ctx': n_ctx, 'max_tokens': max_tokens, 'n_gpu_layers': -1, 'n_batch': 128, 'n_threads': 4, 'n_threads_batch': 4},
                    'system_prompt': SYSTEM, 'memory_before': before}
        stop = threading.Event()
        critical = threading.Event()
        samples = []
        def monitor():
            while not stop.wait(2):
                try:
                    state = memory_state()
                except Exception as exc:
                    samples.append({'elapsed_seconds': time.perf_counter() - started, 'monitor_error': str(exc)})
                    critical.set()
                    return
                samples.append({'elapsed_seconds': time.perf_counter() - started, **state})
                if state['pressure_level'] == 4 or (state['free_percent'] is not None and state['free_percent'] < 8):
                    critical.set()
        started = time.perf_counter()
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        model = None
        try:
            model = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=-1, n_batch=128, n_threads=4, n_threads_batch=4,
                          seed=seed, verbose=True, use_mmap=True, no_perf=False)
            metadata['constructor_load_seconds'] = time.perf_counter() - started
            metadata['load_seconds'] = metadata['constructor_load_seconds'] + backend_init_seconds
            metadata['chat_format'] = model.chat_format
            metadata['embedded_chat_template'] = model.metadata.get('tokenizer.chat_template')
            for index in indexes:
                check_memory()
                if critical.is_set():
                    raise RuntimeError('Critical memory pressure detected; stopping inference')
                row = records[index]
                model.reset()
                llama_cpp.llama_perf_context_reset(model.ctx)
                begin = time.perf_counter()
                # Streaming lets us stop on memory pressure between generated tokens.
                chunks = model.create_chat_completion(messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row[INPUT]}],
                    temperature=0, top_p=1, top_k=1, repeat_penalty=1, max_tokens=max_tokens, stream=True)
                parts, token_times, finish = [], [], None
                for chunk in chunks:
                    if critical.is_set():
                        chunks.close()
                        raise RuntimeError('Critical memory pressure detected; generation cancelled')
                    choice = chunk['choices'][0]
                    content = choice['delta'].get('content')
                    if content:
                        parts.append(content)
                        token_times.append(time.perf_counter())
                    finish = choice.get('finish_reason') or finish
                elapsed = time.perf_counter() - begin
                text = ''.join(parts)
                # llama.cpp counters measure actual evaluated tokens, not text chunks.
                timings = llama_cpp.llama_perf_context(model.ctx)
                completion_tokens = int(timings.n_eval)
                generation_seconds = float(timings.t_eval_ms) / 1000
                result = {'dataset_index': index, 'input': row[INPUT], 'reference': row[REFERENCE], 'output': text,
                          'finish_reason': finish, 'wall_seconds': elapsed,
                          'time_to_first_text_seconds': token_times[0] - begin if token_times else None,
                          'prompt_eval_tokens': int(timings.n_p_eval), 'prompt_eval_seconds': float(timings.t_p_eval_ms) / 1000,
                          'completion_eval_tokens': completion_tokens, 'generation_eval_seconds': generation_seconds,
                          'generation_tokens_per_second': completion_tokens / generation_seconds if generation_seconds else None,
                          'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          **compare(text, row[REFERENCE])}
                dest.write(json.dumps(result, ensure_ascii=False) + '\n')
                dest.flush()
                print(f'row {index}: {elapsed:.1f}s, {completion_tokens} evaluated tokens, {finish}', flush=True)
        except BaseException as exc:
            metadata['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if model is not None:
                model.close()
            stop.set()
            watcher.join(timeout=5)
            metadata['total_seconds'] = time.perf_counter() - started
            metadata['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metadata['memory_samples'] = samples
            metadata['memory_after_close'] = memory_state()
            output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/dataset.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/baseline.jsonl')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--n-ctx', type=int, default=2048)
    run(**vars(parser.parse_args()))
