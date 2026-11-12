"""Independent trainable PyTorch Llama decoder, initialized directly from F16 GGUF.

No Transformers attention or decoder implementation is called by this model.
GGUF rotary pairs are adjacent (the converter has already permuted Q and K).
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DecodedTransformer(nn.Module):
    def __init__(self, path):
        super().__init__()
        from gguf import GGUFReader
        reader = GGUFReader(path, 'r')
        if reader.fields['general.architecture'].contents() != 'llama':
            raise ValueError('Own decoder supports the controlled SmolLM2 Llama architecture')
        self.config = {k: f.contents() for k, f in reader.fields.items() if k.startswith('llama.')}
        self.weights = nn.ParameterDict()
        for tensor in reader.tensors:
            if tensor.tensor_type.name not in ('F16', 'F32'):
                raise ValueError('Decode an F16/F32 GGUF, not packed quantization')
            self.weights[tensor.name.replace('.', '__')] = nn.Parameter(
                torch.from_numpy(tensor.data.astype(np.float32).copy()))
        self.heads = self.config['llama.attention.head_count']
        self.kv_heads = self.config['llama.attention.head_count_kv']
        self.width = self.config['llama.embedding_length']
        self.dim = self.width // self.heads

    def w(self, name):
        return self.weights[name.replace('.', '__') + '__weight']

    def norm(self, x, name):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.config['llama.attention.layer_norm_rms_epsilon'])).to(x.dtype) * self.w(name)

    def block(self, x, cos, sin, layer, cache=None, return_cache=False):
        length = x.shape[1]
        offset = cache[0].shape[1] if cache else 0
        p = f'blk.{layer}.'

        def rope(z):
            a, b = z[..., 0::2], z[..., 1::2]
            return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

        z = self.norm(x, p + 'attn_norm')
        q = rope(F.linear(z, self.w(p + 'attn_q')).reshape(1, length, self.heads, self.dim))
        k = rope(F.linear(z, self.w(p + 'attn_k')).reshape(1, length, self.kv_heads, self.dim))
        v = F.linear(z, self.w(p + 'attn_v')).reshape(1, length, self.kv_heads, self.dim)
        if cache:
            k = torch.cat((cache[0], k), dim=1)
            v = torch.cat((cache[1], v), dim=1)
        next_cache = (k, v) if return_cache else None
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=2)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=2)
        # Explicit math allows float32 parity diagnostics against the NumPy oracle.
        scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(self.dim)
        mask = (torch.arange(offset + length, device=x.device)[None, :] >
                torch.arange(offset, offset + length, device=x.device)[:, None])
        probs = scores.masked_fill(mask, float('-inf')).float().softmax(-1).to(x.dtype)
        attended = (probs @ v.transpose(1, 2)).transpose(1, 2).reshape(1, length, self.width)
        x = x + F.linear(attended, self.w(p + 'attn_output'))
        z = self.norm(x, p + 'ffn_norm')
        x = x + F.linear(F.silu(F.linear(z, self.w(p + 'ffn_gate'))) *
                         F.linear(z, self.w(p + 'ffn_up')), self.w(p + 'ffn_down'))
        return x, next_cache

    def forward(self, tokens, *, cache=None, return_cache=False):
        from torch.utils.checkpoint import checkpoint
        if tokens.shape[0] != 1:
            raise ValueError('The independent decoder supports one sequence per batch')
        x = F.embedding(tokens, self.w('token_embd'))
        length = tokens.shape[1]
        offset = cache[0][0].shape[1] if cache else 0
        next_cache = []
        angles = torch.arange(offset, offset + length, device=x.device, dtype=torch.float32)[:, None] * (
            self.config['llama.rope.freq_base'] **
            (-torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim))[None, :]
        cos, sin = angles.cos().to(x.dtype)[None, :, None], angles.sin().to(x.dtype)[None, :, None]
        for layer in range(self.config['llama.block_count']):
            if self.training and torch.is_grad_enabled():
                if cache or return_cache:
                    raise ValueError('Training does not retain a KV cache')
                # Recompute block activations on backward so full 2048-token targets fit Colab GPUs.
                x, _ = checkpoint(self.block, x, cos, sin, layer, use_reentrant=False)
            else:
                x, saved = self.block(x, cos, sin, layer, cache[layer] if cache else None, return_cache)
                if return_cache:
                    next_cache.append(saved)
        output = self.w('output') if 'output__weight' in self.weights else self.w('token_embd')
        logits = F.linear(self.norm(x, 'output_norm'), output)
        return (logits, next_cache) if return_cache else logits


def parity(model, gguf_path, run):
    """Retain the original tolerance and report failures without relaxing it."""
    from llama_cpp import Llama
    reference = Llama(model_path=str(gguf_path), n_ctx=128, n_batch=128,
                      n_gpu_layers=0, logits_all=True, verbose=False)
    try:
        tokens = reference.tokenize(b'The quiet bridge beside the river.', add_bos=True)[:5]
        reference.eval(tokens)
        expected = reference.scores[:len(tokens)].copy()
    finally:
        reference.close()
    model.eval()
    with torch.no_grad():
        actual = model(torch.tensor([tokens], device=next(model.parameters()).device)).float().cpu().numpy()[0]
    error = np.abs(actual.astype(np.float64) - expected)
    acceptance = dict(atol=0.001, rtol=0.001)
    report = dict(token_ids=tokens, acceptance=acceptance,
                  max_absolute_error=float(error.max()), rms_error=float(np.sqrt((error ** 2).mean())),
                  violating_logits=int((error > acceptance['atol'] + acceptance['rtol'] * np.abs(expected)).sum()),
                  argmax_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
                  criterion_passed=bool(np.allclose(actual, expected, **acceptance)),
                  interpretation='Training is exploratory if parity fails. CPU GGML operand rounding, reductions and kernel ordering differ from float32 PyTorch. This does not establish exact llama.cpp equivalence.')
    np.savez_compressed(run.path / 'parity-logits.npz', actual=actual, expected=expected, tokens=tokens)
    run.json('parity.json', report)
    return report
