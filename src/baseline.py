"""Text-only, bounded local baseline. Each output row is durable before the next input."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.metadata
import json
import os
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


def check_memory(state=None):
    state = memory_state() if state is None else state
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


def run(dataset=ROOT / 'data/dataset.json', output=ROOT / 'results/01/local-2026-11-11/baseline.jsonl', limit=10, seed=42, max_tokens=1024, n_ctx=2048, resume=False, max_new_rows=None, n_gpu_layers=-1):
    if not 1 <= limit <= 150 or not 1 <= max_tokens < n_ctx:
        raise ValueError('Require 1 <= limit <= 150 and 1 <= max_tokens < n_ctx')
    if max_new_rows is not None and max_new_rows < 1:
        raise ValueError('max_new_rows must be positive')
    if n_gpu_layers not in (-1, 0):
        raise ValueError('n_gpu_layers must be -1 (Metal) or 0 (CPU)')
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
    with (ROOT / 'models/.inference.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((ROOT / 'models/manifest.json').read_text())
        model_path = ROOT / 'models' / manifest['files'][0]['name']
        settings = {'seed': seed, 'temperature': 0, 'top_p': 1, 'top_k': 1, 'repeat_penalty': 1,
                    'n_ctx': n_ctx, 'max_tokens': max_tokens, 'n_gpu_layers': n_gpu_layers,
                    'n_batch': 128, 'n_threads': 4, 'n_threads_batch': 4}
        provenance = {'dataset_sha256': hashlib.sha256(data_bytes).hexdigest(),
                      'model': manifest, 'settings': settings, 'system_prompt': SYSTEM}
        existing = []
        if output.exists():
            if not resume:
                raise FileExistsError(output)
            raw = output.read_bytes()
            if raw and not raw.endswith(b'\n'):
                raise ValueError('Incomplete trailing JSONL row; preserve and inspect before resuming')
            existing = [json.loads(line) for line in raw.splitlines()]
            prior_paths = [output.with_suffix('.metadata.json'), *sorted(output.parent.glob(output.stem + '.attempt-*.metadata.json'))]
            prior = [json.loads(path.read_text()) for path in prior_paths if path.exists()]
            if not prior or any(any(meta.get(k) != v for k, v in provenance.items()) for meta in prior):
                raise ValueError('Resume provenance mismatch or missing metadata')
            seen = set()
            for row in existing:
                index = row['dataset_index']
                if type(index) is not int or not 0 <= index < len(records) or index in seen:
                    raise ValueError('Invalid or duplicate saved dataset index')
                if row['input'] != records[index][INPUT] or row['reference'] != records[index][REFERENCE] or not row.get('finish_reason'):
                    raise ValueError('Saved row does not match dataset or is unfinished')
                seen.add(index)
        else:
            seen = set()
        pending = [index for index in indexes if index not in seen]
        if max_new_rows is not None:
            pending = pending[:max_new_rows]
        if not pending:
            print('All selected indexes already complete; no model loaded.', flush=True)
            return output
        # Verify actual immutable model bytes before associating new evidence with the manifest.
        with model_path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != manifest['files'][0]['sha256'] or model_path.stat().st_size != manifest['files'][0]['size_bytes']:
            raise ValueError('Model verification failed')
        attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        metadata_path = output.with_name(f'{output.stem}.attempt-{attempt_id}.metadata.json')
        metadata = {**provenance, 'attempt_id': attempt_id, 'sample_indexes': indexes,
                    'pending_indexes': pending, 'previously_completed_indexes': sorted(seen), 'completed_indexes': []}
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        try:
            before = memory_state()
            metadata['memory_before'] = before
            check_memory(before)
        except BaseException as exc:
            metadata['error'] = f'{type(exc).__name__}: {exc}'
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
            raise
        backend_started = time.perf_counter()
        from llama_cpp import Llama, llama_cpp
        system_info = llama_cpp.llama_print_system_info().decode()
        backend_init_seconds = time.perf_counter() - backend_started
        metadata.update({'started_utc': datetime.now(timezone.utc).isoformat(), 'dataset_sha256': hashlib.sha256(data_bytes).hexdigest(),
                    'dataset_rows': len(records), 'sample_indexes': indexes, 'model': manifest,
                    'python': platform.python_version(), 'macos': platform.mac_ver()[0], 'machine': platform.machine(),
                    'llama_cpp_python': importlib.metadata.version('llama-cpp-python'),
                    'llama_system_info': system_info, 'backend_init_seconds': backend_init_seconds,
                    'settings': settings,
                    'system_prompt': SYSTEM, 'memory_before': before})
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
        dest = None
        try:
            model = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_batch=128, n_threads=4, n_threads_batch=4,
                          seed=seed, verbose=True, use_mmap=True, no_perf=False)
            metadata['constructor_load_seconds'] = time.perf_counter() - started
            metadata['load_seconds'] = metadata['constructor_load_seconds'] + backend_init_seconds
            metadata['chat_format'] = model.chat_format
            metadata['embedded_chat_template'] = model.metadata.get('tokenizer.chat_template')
            dest = output.open('a' if resume else 'x')
            for index in pending:
                metadata['active_dataset_index'] = index
                state = memory_state()
                samples.append({'elapsed_seconds': time.perf_counter() - started, 'dataset_index': index, 'phase': 'before_input', **state})
                check_memory(state)
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
                result = {'attempt_id': attempt_id, 'dataset_index': index, 'input': row[INPUT], 'reference': row[REFERENCE], 'output': text,
                          'finish_reason': finish, 'wall_seconds': elapsed,
                          'time_to_first_text_seconds': token_times[0] - begin if token_times else None,
                          'prompt_eval_tokens': int(timings.n_p_eval), 'prompt_eval_seconds': float(timings.t_p_eval_ms) / 1000,
                          'completion_eval_tokens': completion_tokens, 'generation_eval_seconds': generation_seconds,
                          'generation_tokens_per_second': completion_tokens / generation_seconds if generation_seconds else None,
                          'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          **compare(text, row[REFERENCE])}
                dest.write(json.dumps(result, ensure_ascii=False) + '\n')
                dest.flush()
                os.fsync(dest.fileno())
                metadata['completed_indexes'].append(index)
                samples.append({'elapsed_seconds': time.perf_counter() - started, 'dataset_index': index, 'phase': 'after_input', **memory_state()})
                print(f'row {index}: {elapsed:.1f}s, {completion_tokens} evaluated tokens, {finish}', flush=True)
        except BaseException as exc:
            metadata['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if dest is not None:
                dest.close()
            if model is not None:
                model.close()
            stop.set()
            watcher.join(timeout=5)
            metadata['total_seconds'] = time.perf_counter() - started
            metadata['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metadata['memory_samples'] = samples
            metadata['memory_after_close'] = memory_state()
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/dataset.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/01/local-2026-11-11/baseline.jsonl')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--n-ctx', type=int, default=2048)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-rows', type=int)
    parser.add_argument('--n-gpu-layers', type=int, choices=[-1, 0], default=-1)
    run(**vars(parser.parse_args()))
