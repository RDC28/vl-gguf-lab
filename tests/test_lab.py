"""Behavioral checks for persistence, scoring, edits, decoder and optimizer recovery."""
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from src.lab_edits import rewrite
from src.lab_experiments import evaluate
from src.lab_metrics import score, structure
from src.lab_storage import Run, results_root
from src.lab_training import train
from src.lab_transformer import DecodedTransformer
from src.freeze_split import build


@pytest.fixture
def tiny_gguf(tmp_path):
    from gguf import GGUFWriter
    path = tmp_path / 'tiny.gguf'
    writer = GGUFWriter(path, 'llama')
    params = {'block_count': 2, 'embedding_length': 8, 'attention.head_count': 2,
              'attention.head_count_kv': 1, 'attention.layer_norm_rms_epsilon': 1e-5,
              'rope.freq_base': 10000.0}
    for name, value in params.items():
        if isinstance(value, float):
            writer.add_float32('llama.' + name, value)
        else:
            writer.add_uint32('llama.' + name, value)
    rng = np.random.default_rng(42)
    arrays = {'token_embd.weight': rng.normal(0, .1, (16, 8)).astype('float32'),
              'output_norm.weight': np.ones(8, dtype='float32')}
    for layer in range(2):
        for name, shape in {'attn_norm': (8,), 'ffn_norm': (8,), 'attn_q': (8, 8),
                            'attn_k': (4, 8), 'attn_v': (4, 8), 'attn_output': (8, 8),
                            'ffn_gate': (12, 8), 'ffn_up': (12, 8), 'ffn_down': (8, 12)}.items():
            arrays[f'blk.{layer}.{name}.weight'] = (np.ones(shape) if len(shape) == 1 else
                                                   rng.normal(0, .1, shape)).astype('float32')
    for name, data in arrays.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return path


def test_storage_resume_and_interrupted_tail(tmp_path):
    run = Run(1, {'seed': 42}, root=tmp_path)
    run.append('output.jsonl', {'index': 0, 'text': 'complete'})
    with (run.path / 'output.jsonl').open('ab') as stream:
        stream.write(b'{"index":1')
    resumed = Run(1, {'seed': 42}, root=tmp_path, resume=run.path.name)
    assert resumed.rows('output.jsonl') == [{'index': 0, 'text': 'complete'}]
    resumed.append('output.jsonl', {'index': 1, 'text': 'recovered'})
    assert len(resumed.rows('output.jsonl')) == 2
    with pytest.raises(ValueError, match='differs'):
        Run(1, {'seed': 7}, root=tmp_path, resume=run.path.name)
    assert results_root(local_root=tmp_path) == tmp_path


def test_progress_log_reaches_console_during_capture(tmp_path, capsys):
    run = Run(1, {}, root=tmp_path)
    with run.capture():
        print('library noise')
        run.log('scene 3 completed')
    console = capsys.readouterr().out
    assert 'scene 3 completed' in console
    assert 'library noise' not in console
    assert 'library noise' in (run.path / 'runtime.log').read_text()
    assert run.rows('events.jsonl')[-1]['message'] == 'scene 3 completed'


def test_metrics_and_evaluation_skip_completed(tmp_path):
    text = 'SHOT 1 (00-02s) 32 mm wide static. SHOT 2 (02-04s) 50 mm medium track.'
    assert structure(text)['shot_count'] == 2
    metrics = score(text, text)
    assert metrics['unigram_f1'] == 1
    assert metrics['structure_score'] == 1
    assert score('', text)['structure_score'] == 0
    run = Run(1, {}, root=tmp_path)
    data = [{"user's scene description": 'bridge', 'shot breakdown': text}]
    calls = []
    def generator(scene):
        calls.append(scene)
        return dict(text=text, truncated=False)
    assert evaluate(generator, data, [0], run, 'baseline')['n'] == 1
    assert evaluate(generator, data, [0], run, 'baseline')['truncation_rate'] == 0
    assert calls == ['bridge']


def test_split_is_disjoint_and_grouped():
    split = build()
    sets = {k: set(v) for k, v in split['indexes'].items()}
    assert len(set.union(*sets.values())) == 150
    assert sum(map(len, sets.values())) == 150
    for group in split['connected_groups']:
        assert sum(bool(set(group) & s) for s in sets.values()) == 1


def test_edit_control_and_shapes(tiny_gguf, tmp_path):
    control = rewrite(tiny_gguf, tmp_path / 'control.gguf', operation='control')
    modified = rewrite(tiny_gguf, tmp_path / 'norm.gguf', operation='norm', layer=1, strength=1.1)
    changed = [k for k in control['tensors'] if control['tensors'][k] != modified['tensors'][k]]
    assert changed == ['blk.1.attn_norm.weight']
    for operation, expected in [('add', 3), ('remove', 1)]:
        path = tmp_path / f'{operation}.gguf'
        result = rewrite(tiny_gguf, path, operation=operation, layer=0)
        model = DecodedTransformer(path)
        assert len(result['block_map']) == expected
        assert model(torch.tensor([[1, 2]])).shape == (1, 2, 16)
    assert torch.equal(DecodedTransformer(tiny_gguf)(torch.tensor([[1, 2]])),
                       DecodedTransformer(tmp_path / 'control.gguf')(torch.tensor([[1, 2]])))


def test_decoder_numpy_parity_cache_and_gradient(tiny_gguf, monkeypatch):
    from gguf import GGUFReader
    # Existing independently written NumPy implementation is the float32 math oracle.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'src'))
    from case6_forward import forward
    model = DecodedTransformer(tiny_gguf).eval()
    tokens = [1, 2, 3, 4]
    reader = GGUFReader(tiny_gguf)
    expected = forward(reader, tokens, model.config, lambda phase: None)
    actual = model(torch.tensor([tokens]))
    np.testing.assert_allclose(actual.detach().numpy()[0], expected, atol=1e-5, rtol=1e-5)
    cache = None
    outputs = []
    for token in tokens:
        output, cache = model(torch.tensor([[token]]), cache=cache, return_cache=True)
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, 1), actual, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


class ToyTokenizer:
    eos_token_id = 0
    def apply_chat_template(self, *args, **kwargs):
        return [1, 2]
    def encode(self, *args, **kwargs):
        return [3, 4, 5]


def test_training_resume_matches_uninterrupted(tiny_gguf, tmp_path):
    config = dict(seed=42, learning_rate=.001, epochs=1, train_steps=None,
                  checkpoint_every=1, drive_snapshot_every=None, device='cpu', train_context=16)
    data = [{"user's scene description": 'bridge', 'shot breakdown': 'reference'}] * 2
    complete = DecodedTransformer(tiny_gguf)
    initial = {k: v.detach().clone() for k, v in complete.named_parameters()}
    interrupted = DecodedTransformer(tiny_gguf)
    full_run = Run(2, config, root=tmp_path / 'full')
    train(complete, ToyTokenizer(), data, [0, 1], config, full_run, own=True)
    assert any(not torch.equal(v, initial[k]) for k, v in complete.named_parameters())
    retry_run = Run(2, config, root=tmp_path / 'retry')
    append = retry_run.append
    def crash(name, value):
        append(name, value)
        if name == 'training.jsonl' and value['step'] == 2:
            raise RuntimeError('simulated disconnect before checkpoint')
    retry_run.append = crash
    with pytest.raises(RuntimeError, match='disconnect'):
        train(interrupted, ToyTokenizer(), data, [0, 1], config, retry_run, own=True)
    retry_run.append = append
    fresh = DecodedTransformer(tiny_gguf)
    train(fresh, ToyTokenizer(), data, [0, 1], config, retry_run, own=True)
    assert [r['step'] for r in retry_run.rows('training.jsonl')] == [1, 2]
    for a, b in zip(complete.parameters(), fresh.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def snapshot_setup(tmp_path, monkeypatch):
    local = tmp_path / 'local-disk'
    local.mkdir()
    monkeypatch.setattr('tempfile.tempdir', str(local))
    config = dict(seed=42, learning_rate=.001, epochs=1, train_steps=None,
                  checkpoint_every=1, drive_snapshot_every=1, device='cpu', train_context=16)
    data = [{"user's scene description": 'bridge', 'shot breakdown': 'reference'}] * 2
    run = Run(6, config, root=tmp_path / 'drive')
    return local / 'vl-gguf-lab-checkpoints' / run.path.name, config, data, run


def test_training_falls_back_to_drive_snapshot_after_runtime_reset(tiny_gguf, tmp_path, monkeypatch):
    local, config, data, run = snapshot_setup(tmp_path, monkeypatch)
    append = run.append
    def crash(name, value):
        append(name, value)
        if name == 'training.jsonl' and value['step'] == 2:
            raise RuntimeError('simulated runtime reset')
    run.append = crash
    with pytest.raises(RuntimeError, match='reset'):
        train(DecodedTransformer(tiny_gguf), ToyTokenizer(), data, [0, 1], config, run, own=True)
    run.append = append
    assert (local / 'checkpoint.pt').exists()
    assert not (run.path / 'checkpoint.pt').exists()
    first = torch.load(run.path / 'weights-snapshot.pt', weights_only=True)
    assert first['step'] == 1
    assert all(v.dtype == torch.float16 for v in first['parameters'].values())
    shutil.rmtree(local)
    resumed = DecodedTransformer(tiny_gguf)
    result = train(resumed, ToyTokenizer(), data, [0, 1], config, run, own=True)
    assert result['optimizer_restarted_at_steps'] == [1]
    assert result['float16_snapshot_restores_at_steps'] == [1]
    assert [r['step'] for r in run.rows('training.jsonl')] == [1, 2]
    final = torch.load(run.path / 'weights-snapshot.pt', weights_only=True)
    assert final['step'] == 2
    for key, value in resumed.named_parameters():
        torch.testing.assert_close(final['parameters'][key].float(), value.detach(), atol=0, rtol=0)


def test_after_evaluation_weights_identical_across_post_training_resumes(tiny_gguf, tmp_path, monkeypatch):
    local, config, data, run = snapshot_setup(tmp_path, monkeypatch)
    tokens = torch.tensor([[1, 2, 3]])
    completed = DecodedTransformer(tiny_gguf)
    result = train(completed, ToyTokenizer(), data, [0, 1], config, run, own=True)
    assert result['after_evaluation_weights'] == 'float16 final Drive snapshot'
    assert result['float16_snapshot_restores_at_steps'] == []
    with torch.no_grad():
        expected = completed(tokens)
    assert (local / 'checkpoint.pt').exists()
    same_runtime = DecodedTransformer(tiny_gguf)
    assert train(same_runtime, ToyTokenizer(), data, [0, 1], config, run, own=True)[
        'float16_snapshot_restores_at_steps'] == []
    shutil.rmtree(local)
    reset_runtime = DecodedTransformer(tiny_gguf)
    result = train(reset_runtime, ToyTokenizer(), data, [0, 1], config, run, own=True)
    assert result['float16_snapshot_restores_at_steps'] == [2]
    assert result['optimizer_restarted_at_steps'] == []
    with torch.no_grad():
        assert torch.equal(same_runtime(tokens), expected)
        assert torch.equal(reset_runtime(tokens), expected)
