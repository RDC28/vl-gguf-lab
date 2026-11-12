"""Response-only supervised training and identical-settings held-out evaluation."""
import gc
from pathlib import Path
import random
import tempfile
import time

import torch
from torch.nn import functional as F

from .lab_models import HF_MODELS, SYSTEM
from .lab_storage import atomic_json


def prompt_tokens(tokenizer, scene):
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': scene}],
        tokenize=True, add_generation_prompt=True)


class TorchGenerator:
    def __init__(self, model, tokenizer, config, *, own=False):
        self.model, self.tokenizer, self.config, self.own = model, tokenizer, config, own

    def __call__(self, scene):
        self.model.eval()
        ids = prompt_tokens(self.tokenizer, scene)
        if len(ids) + self.config['max_tokens'] > self.config['context']:
            raise ValueError('Prompt plus generation cap exceeds context')
        device = next(self.model.parameters()).device
        eos = self.tokenizer.eos_token_id
        with torch.no_grad():
            tokens = torch.tensor([ids], device=device)
            if self.own:
                generated, cache = [], None
                for _ in range(self.config['max_tokens']):
                    logits, cache = self.model(tokens, cache=cache, return_cache=True)
                    next_id = int(logits[0, -1].argmax())
                    generated.append(next_id)
                    if next_id == eos:
                        break
                    tokens = torch.tensor([[next_id]], device=device)
            else:
                output = self.model.generate(tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
                                             max_new_tokens=self.config['max_tokens'],
                                             pad_token_id=eos, eos_token_id=eos,
                                             use_cache=True)
                generated = output[0, len(ids):].tolist()
        truncated = len(generated) == self.config['max_tokens'] and generated[-1] != eos
        return dict(text=self.tokenizer.decode(generated, skip_special_tokens=True), truncated=truncated,
                    finish_reason='length' if truncated else 'stop',
                    usage=dict(prompt_tokens=len(ids), completion_tokens=len(generated)))


def load_hf(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    repo, revision = HF_MODELS[config['training_tier']]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    base = AutoModelForCausalLM.from_pretrained(repo, revision=revision, torch_dtype=torch.float32,
                                               attn_implementation='sdpa' if config['device'] == 'cuda' else 'eager').to(config['device'])
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                                           target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    if config['device'] == 'cuda':
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    return model, tokenizer


def save(path, value):
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def train(model, tokenizer, data, indexes, config, run, *, own=False):
    """Fixed epoch budget, train split only, exact optimizer/RNG resumption from full checkpoints."""
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=config['learning_rate'])
    snapshot_every = config['drive_snapshot_every']
    if snapshot_every:
        checkpoint = Path(tempfile.gettempdir()) / 'vl-gguf-lab-checkpoints' / run.path.name / 'checkpoint.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint = run.path / 'checkpoint.pt'
    snapshot = run.path / 'weights-snapshot.pt'
    schedule = []
    rng = random.Random(config['seed'])
    for _ in range(config['epochs']):
        epoch = list(indexes)
        rng.shuffle(epoch)
        schedule.extend(epoch)
    if config['train_steps']:
        schedule = schedule[:config['train_steps']]
    start_step = 0
    history = []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for key, value in state['parameters'].items():
                trainable[key].copy_(value.to(trainable[key].device))
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        if config['device'] == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_step = state['step']
        history = state['history']
    elif snapshot_every and snapshot.exists():
        state = torch.load(snapshot, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for key, value in state['parameters'].items():
                trainable[key].copy_(value.to(trainable[key].device))
        start_step = state['step']
        history = state['history']
        run.append('snapshot-restores.jsonl', dict(step=start_step))
        run.log(f'Restored float16 weights from Drive snapshot at step {start_step}; optimizer state not recovered')
    # Remove uncommitted log rows after the last durable optimizer checkpoint.
    log_path = run.path / 'training.jsonl'
    log_path.write_text('')
    for row in history:
        run.append('training.jsonl', row)
    model.train()
    for step in range(start_step, len(schedule)):
        row = data[schedule[step]]
        prompt = prompt_tokens(tokenizer, row["user's scene description"])
        target = tokenizer.encode(row['shot breakdown'], add_special_tokens=False) + [tokenizer.eos_token_id]
        full = prompt + target
        original_length = len(full)
        full = full[:config['train_context']]
        if len(full) <= len(prompt):
            raise ValueError('Training context leaves no response tokens; increase train_context')
        input_ids = torch.tensor([full[:-1]], device=config['device'])
        labels = torch.tensor([full[1:]], device=config['device'])
        labels[:, :len(prompt) - 1] = -100
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids) if own else model(input_ids, use_cache=False).logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = dict(step=step + 1, dataset_index=schedule[step], loss=float(loss.detach()),
                      gradient_norm=float(norm), supervised_tokens=len(full) - len(prompt),
                      sequence_truncated=original_length > len(full), seconds=time.perf_counter() - begin)
        history.append(record)
        run.append('training.jsonl', record)
        del logits, loss
        last = step + 1 == len(schedule)
        if snapshot_every and ((step + 1) % snapshot_every == 0 or last):
            save(snapshot, dict(step=step + 1, history=history,
                                parameters={k: p.detach().half().cpu() for k, p in trainable.items()}))
            run.log(f'Drive weight snapshot saved at step {step + 1}')
        if (step + 1) % config['checkpoint_every'] == 0 or last:
            save(checkpoint, dict(step=step + 1, history=history,
                                  parameters={k: p.detach().cpu() for k, p in trainable.items()},
                                  optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                                  cuda_rng=torch.cuda.get_rng_state_all() if config['device'] == 'cuda' else None))
            run.log(f'Checkpoint durable at step {step + 1}')
    if snapshot_every:
        with torch.no_grad():
            for p in trainable.values():
                p.copy_(p.half())
    history = run.rows('training.jsonl')
    restores = [r['step'] for r in run.rows('snapshot-restores.jsonl')]
    result = dict(steps=len(history), trainable_parameters=sum(p.numel() for p in trainable.values()),
                  total_supervised_tokens=sum(r['supervised_tokens'] for r in history),
                  sequence_truncation_rate=sum(r['sequence_truncated'] for r in history) / max(1, len(history)),
                  epochs=config['epochs'], schedule_size=len(schedule), checkpoint_every=config['checkpoint_every'],
                  drive_snapshot_every=snapshot_every,
                  after_evaluation_weights='float16 final Drive snapshot' if snapshot_every else 'float32',
                  float16_snapshot_restores_at_steps=restores,
                  optimizer_restarted_at_steps=[step for step in restores if step < len(schedule)])
    atomic_json(run.path / 'training-summary.json', result)
    model.eval()
    gc.collect()
    if config['device'] == 'cuda':
        torch.cuda.empty_cache()
    return result
