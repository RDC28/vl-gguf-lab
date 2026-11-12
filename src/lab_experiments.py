"""Six thin-notebook entry points, sharing data, scoring and persistent storage."""
import gc
import json
import os
from pathlib import Path
import tempfile
import time

from .freeze_split import build
from .lab_metrics import aggregate, score
from .lab_models import CATALOG, HF_MODELS, GGUFGenerator, download
from .lab_storage import ROOT, Run, digest


def configuration(case, smoke=False):
    return dict(case=case, smoke=smoke, seed=42,
                model_tier='small' if smoke else 't4', training_tier='small' if smoke else 'train',
                device='cpu' if smoke else 'cuda', gpu_layers=0 if smoke else -1,
                context=4096 if smoke else 8192, max_tokens=24 if smoke else 4096,
                train_context=256 if smoke else 2048, epochs=1 if smoke else 3,
                train_steps=1 if smoke else None, checkpoint_every=1 if smoke else 10,
                drive_snapshot_every=(1 if smoke else 120) if case == 6 else None,
                learning_rate=0.0001, limit=1 if smoke else None,
                norm_strengths=[1.01] if smoke else [0.9, 1.01, 1.1],
                edit_layers=[15] if smoke else [0, 15, 29],
                drive_root='MyDrive/vl-gguf-lab/results')


def evaluate(generator, data, indexes, run, name):
    saved = run.rows(name + '.jsonl')
    by_index = {row['index']: row for row in saved}
    if len(by_index) != len(saved) or not set(by_index).issubset(indexes):
        raise ValueError('Duplicate or unexpected scene in resumed evaluation')
    for index in indexes:
        if index in by_index:
            continue
        scene = data[index]
        begin = time.perf_counter()
        prediction = generator(scene["user's scene description"])
        row = dict(index=index, input=scene["user's scene description"], reference=scene['shot breakdown'],
                   **prediction, seconds=time.perf_counter() - begin,
                   scores=score(prediction['text'], scene['shot breakdown']))
        run.append(name + '.jsonl', row)
        by_index[index] = row
        run.json(name + '-metrics.json', aggregate(list(by_index.values())))
        run.log(f'{name}: completed scene {index} ({len(by_index)}/{len(indexes)})')
    return aggregate([by_index[i] for i in indexes])


def paired(run, variant, baseline='baseline'):
    import numpy as np
    a = {r['index']: r for r in run.rows(baseline + '.jsonl')}
    b = {r['index']: r for r in run.rows(variant + '.jsonl')}
    if set(a) != set(b):
        raise ValueError('Paired evaluations have different scenes')
    result = {}
    for metric in ('unigram_f1', 'structure_score'):
        delta = np.array([b[i]['scores'][metric] - a[i]['scores'][metric] for i in sorted(a)])
        bootstrap = np.random.default_rng(42).choice(delta, size=(1000, len(delta)), replace=True).mean(1)
        result[metric] = dict(mean_delta=float(delta.mean()),
                              scene_bootstrap_95_interval=np.quantile(bootstrap, [0.025, 0.975]).tolist())
    return result


def run_case(case, config, *, resume=None, root=None):
    import torch
    if config['case'] != case:
        raise ValueError('Config case differs from notebook case')
    if not config['smoke'] and config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime or set SMOKE=True for the CPU check')
    torch.manual_seed(config['seed'])
    if config['smoke']:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    data_path = ROOT / 'data/dataset.json'
    data = json.loads(data_path.read_text())
    split = build()
    config = dict(config, dataset_sha256=digest(data_path), catalog=CATALOG,
                  hf_models=HF_MODELS, split_sha256=split['split_sha256'],
                  implementation={p.name: digest(p) for p in sorted((ROOT / 'src').glob('lab_*.py'))})
    # JSON normalization gives tuples the same representation on first run and resume.
    config = json.loads(json.dumps(config))
    run = Run(case, config, root=root, resume=resume)
    run.json('split.json', split)
    run.log('Experiment started')
    indexes = list(range(len(data))) if case == 1 else split['indexes']['test']
    if config['limit']:
        indexes = indexes[:config['limit']]
    run.json('evaluation-indexes.json', indexes)
    summaries, comparisons = {}, {}
    try:
        with run.capture():
            if case == 1:
                model_path = download(config['model_tier'])
                run.json('model.json', dict(tier=config['model_tier'], sha256=digest(model_path)))
                generator = GGUFGenerator(model_path, config)
                try:
                    summaries['baseline'] = evaluate(generator, data, indexes, run, 'baseline')
                finally:
                    generator.close()
            elif case in (3, 4, 5):
                from .lab_edits import rewrite
                source = download('small')
                run.json('model.json', dict(tier='small', sha256=digest(source)))
                generator = GGUFGenerator(source, config)
                try:
                    summaries['baseline'] = evaluate(generator, data, indexes, run, 'baseline')
                finally:
                    generator.close()
                variants = [('control', 'control', 0, 1.0)]
                for layer in config['edit_layers']:
                    if case == 3:
                        variants += [(f'norm-L{layer}-{strength}', 'norm', layer, strength)
                                     for strength in config['norm_strengths']]
                    else:
                        op = 'add' if case == 4 else 'remove'
                        variants.append((f'{op}-L{layer}', op, layer, 1.0))
                for name, operation, layer, strength in variants:
                    with tempfile.TemporaryDirectory(prefix='vlgguf-edit-') as temp:
                        target = Path(temp) / 'variant.gguf'
                        audit = rewrite(source, target, operation=operation, layer=layer, strength=strength)
                        run.json(name + '-edit.json', audit)
                        generator = GGUFGenerator(target, config)
                        try:
                            summaries[name] = evaluate(generator, data, indexes, run, name)
                        finally:
                            generator.close()
                        comparisons[name] = paired(run, name)
                        if name == 'control':
                            a = run.rows('baseline.jsonl')
                            b = run.rows('control.jsonl')
                            identical = {r['index']: r['text'] for r in a} == {r['index']: r['text'] for r in b}
                            run.json('control-check.json', dict(text_identical=identical))
                            if not identical:
                                raise RuntimeError('No-change rewrite changed greedy output; inspect control before proceeding')
            elif case in (2, 6):
                from .lab_training import TorchGenerator, load_hf, train
                if case == 2:
                    model, tokenizer = load_hf(config)
                else:
                    from transformers import AutoTokenizer
                    from .lab_transformer import DecodedTransformer, parity
                    repo, revision = HF_MODELS['small']
                    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
                    source = download('small')
                    run.json('model.json', dict(tier='small', sha256=digest(source)))
                    model = DecodedTransformer(source)  # parity always float32, CPU versus CPU
                    report = parity(model, source, run)
                    run.log('Parity passed: ' + str(report['criterion_passed']))
                    model.to(config['device'])
                generator = TorchGenerator(model, tokenizer, config, own=case == 6)
                summaries['before'] = evaluate(generator, data, indexes, run, 'before')
                train_indexes = split['indexes']['train']
                train(model, tokenizer, data, train_indexes, config, run, own=case == 6)
                summaries['after'] = evaluate(generator, data, indexes, run, 'after')
                comparisons['after'] = paired(run, 'after', baseline='before')
                del generator, model
                gc.collect()
            else:
                raise ValueError('Unknown case')
        result = dict(case=case, smoke=config['smoke'], summaries=summaries, paired=comparisons,
                      status='completed', path=str(run.path), n_expected=len(indexes))
        if case == 6:
            result['parity'] = json.loads((run.path / 'parity.json').read_text())
        (run.path / 'failure.json').unlink(missing_ok=True)
        run.json('summary.json', result)
        run.log('Experiment completed')
        return result
    except BaseException as exc:
        run.json('failure.json', dict(error=type(exc).__name__, message=str(exc)))
        run.log('Experiment failed: ' + str(exc))
        raise


def show_summary(result):
    import matplotlib.pyplot as plt
    summaries = result['summaries']
    names = list(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(axes, ['unigram_f1', 'structure_score', 'truncation_rate'],
                                ['Word overlap F1', 'Shot structure proxy', 'Truncation rate']):
        ax.bar(names, [summaries[n][metric] for n in names], color='#387ca3')
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=70)
    fig.tight_layout()
    fig.savefig(Path(result['path']) / 'metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    training_log = Path(result['path']) / 'training.jsonl'
    if training_log.exists():
        steps = [json.loads(line) for line in training_log.read_text().splitlines()]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot([r['step'] for r in steps], [r['loss'] for r in steps], color='#387ca3')
        ax.set(xlabel='Optimizer step', ylabel='Response token cross entropy', title='Training loss')
        fig.tight_layout()
        fig.savefig(Path(result['path']) / 'training-loss.png', dpi=150)
        plt.show()
        details = json.loads((Path(result['path']) / 'training-summary.json').read_text())
        print(f"Training: {details['steps']} steps, {details['total_supervised_tokens']} supervised tokens; "
              f"{details['sequence_truncation_rate']:.1%} of training sequences clipped.")
    if result['smoke']:
        print('CPU smoke check only. These limited outputs are not experiment results.')
    print(f"Completed case {result['case']} on {result['n_expected']} scenes per variant.")
    for name, values in summaries.items():
        print(f"{name}: word F1 {values['unigram_f1']:.3f}, structure {values['structure_score']:.3f}, "
              f"truncated {values['truncation_rate']:.1%}.")
    if 'parity' in result:
        print('Strict reference parity:', 'passed' if result['parity']['criterion_passed'] else
              'FAILED. Training is exploratory; numerical equivalence has not been established.')
    print('Inspect raw answers and field parses. Overlap and structure do not establish cinematic quality.')
    print('Saved config, metrics, outputs, logs and any checkpoints:', result['path'])
