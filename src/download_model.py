"""Download only the two pinned artifacts into models, verifying SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    for item in manifest['files']:
        target = ROOT / 'models' / item['name']
        partial = target.with_suffix('.gguf.part')
        if not target.exists():
            url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{item['name']}"
            print(f"Downloading {item['name']}", flush=True)
            # No second model copy in a shared Hugging Face cache.
            with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as dest:
                while chunk := source.read(8 * 1024 * 1024):
                    dest.write(chunk)
            partial.rename(target)
        with target.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if target.stat().st_size != item['size_bytes'] or digest != item['sha256']:
            raise ValueError(f'Artifact verification failed: {target}')
        print(f'Verified {target.name}: {digest}', flush=True)


if __name__ == '__main__':
    main()
