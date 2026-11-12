"""Install pinned notebook dependencies and verify actual CUDA backend support."""
from pathlib import Path
import importlib.metadata
import subprocess
import sys


def setup(*, smoke=False):
    try:
        import google.colab  # noqa: F401
        colab = True
    except ImportError:
        colab = False
    if colab:
        root = Path(__file__).resolve().parents[1]
        imported = {name: sys.modules[name].__version__ for name in ('numpy', 'torch') if name in sys.modules}
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root / 'requirements.txt')], check=True)
        stale = [name for name, version in imported.items() if importlib.metadata.version(name) != version]
        if stale:
            raise RuntimeError(', '.join(stale) + ' changed version after being imported. '
                               'Restart the Colab session and run from the top.')
        if not smoke:
            subprocess.run(['nvidia-smi'], check=True)
            # Official CUDA 12.4 wheel; --only-binary prevents a silent CPU source build.
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', '--force-reinstall',
                            '--no-deps', '--only-binary=:all:', 'llama-cpp-python==0.3.36',
                            '--extra-index-url', 'https://abetlen.github.io/llama-cpp-python/whl/cu124'], check=True)
        else:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--only-binary=:all:',
                            'llama-cpp-python==0.3.36', '--extra-index-url',
                            'https://abetlen.github.io/llama-cpp-python/whl/cpu'], check=True)
    import torch
    import llama_cpp
    print('Python:', sys.version.split()[0], '| PyTorch:', torch.__version__)
    print('CUDA:', torch.cuda.is_available(), '| llama.cpp offload:', llama_cpp.llama_supports_gpu_offload())
    if not smoke and (not torch.cuda.is_available() or not llama_cpp.llama_supports_gpu_offload()):
        raise RuntimeError('A CUDA GPU and CUDA llama.cpp build are required. Choose a GPU runtime, '
                           'then restart the session if dependencies were already imported.')
