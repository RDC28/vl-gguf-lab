"""Verified model catalog and portable GGUF inference."""
from pathlib import Path

VL_REPO = 'unsloth/Qwen2.5-VL-7B-Instruct-GGUF'
VL_REVISION = '68bb8bc4b7df5289c143aaec0ab477a7d4051aab'
SMALL_REPO = 'bartowski/SmolLM2-135M-Instruct-GGUF'
SMALL_REVISION = '09816acd5d99df7be770d85ea30822623dab342c'
CATALOG = {
    't4': (VL_REPO, VL_REVISION, 'Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf'),
    'l4': (VL_REPO, VL_REVISION, 'Qwen2.5-VL-7B-Instruct-Q8_0.gguf'),
    'a100': (VL_REPO, VL_REVISION, 'Qwen2.5-VL-7B-Instruct-BF16.gguf'),
    'small': (SMALL_REPO, SMALL_REVISION, 'SmolLM2-135M-Instruct-f16.gguf'),
}
HF_MODELS = {
    'small': ('HuggingFaceTB/SmolLM2-135M-Instruct', '12fd25f77366fa6b3b4b768ec3050bf629380bac'),
    'train': ('HuggingFaceTB/SmolLM2-1.7B-Instruct', '31b70e2e869a7173562077fd711b654946d38674'),
}
SYSTEM = ('Turn the scene into a cinematic shot breakdown. Include numbered SHOT headings, '
          'time ranges, lens in mm, framing, camera movement, action, continuity, and sound. '
          'Respect the requested duration and character/prop locks.')


def download(tier):
    from huggingface_hub import hf_hub_download
    repo, revision, filename = CATALOG[tier]
    return Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename))


class GGUFGenerator:
    def __init__(self, path, config):
        import llama_cpp
        if config['gpu_layers'] != 0 and not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError('GPU offload unavailable. Re-run CUDA setup or select smoke mode.')
        self.model = llama_cpp.Llama(model_path=str(path), n_ctx=config['context'],
                                    n_gpu_layers=config['gpu_layers'], seed=config['seed'],
                                    n_batch=256, n_threads=4 if config['smoke'] else None, verbose=False)
        self.config = config

    def __call__(self, scene):
        response = self.model.create_chat_completion(
            messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': scene}],
            max_tokens=self.config['max_tokens'], temperature=0, top_p=1)
        choice = response['choices'][0]
        return dict(text=choice['message']['content'] or '', truncated=choice['finish_reason'] == 'length',
                    finish_reason=choice['finish_reason'], usage=response['usage'])

    def close(self):
        self.model.close()
