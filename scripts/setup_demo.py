"""Download public inference dependencies; never downloads research training code."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENCODER_COMMIT = '29a2a31d6a56f677bf450c3be7cdaef423fb7018'
DATA_COMMIT = 'b1499eb9393b12f1499a983f8fedfe7d14869024'
ENCODER_URL = f'https://raw.githubusercontent.com/iLearn-Lab/AAAI25-ENCODER/{ENCODER_COMMIT}'
DATA_URL = f'https://raw.githubusercontent.com/jinyuxu-whut/CIRThan/{DATA_COMMIT}'
CHECKPOINT_SHA = 'a57b84ad54abcfc5923f1e42cfa36303843a30a72572e49f5f033ffe5c923878'
CLIP_SHA = '1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, destination, sha=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (sha is None or digest(destination) == sha):
        print(f'Already present: {destination.relative_to(ROOT)}', flush=True)
        return
    temporary = destination.with_name(destination.name + '.part')
    print(f'Downloading: {destination.relative_to(ROOT)}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'ENCODER-Research-demo'})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output)
    if sha and digest(temporary) != sha:
        raise RuntimeError(f'SHA-256 mismatch: {temporary}. Original file was not replaced.')
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-from', type=Path, help='Reuse weights from an existing local project (hash-checked).')
    args = parser.parse_args()
    vendor = ROOT / 'ENCODER_repo_new'
    for name, sha in [
        ('model_try2.py', 'f0c435dc3d44d1bfb3681606369fd1cc350ab6be91ca4b111f83c5adff4ef4bf'),
        ('token_wise_matching.py', '9416fdb4d187f4f178927f44646dd8ffbcad9c747b3b75ca5eebdc382315708a'),
    ]:
        # Keep verified upstream source separately from the runtime-adapted module.
        source = vendor / (name + '.upstream')
        download(f'{ENCODER_URL}/{name}', source, sha)
        code = source.read_text()
        if name == 'model_try2.py':
            replacements = {
                "clip_path = './'": "clip_path = os.environ['ENCODER_CLIP_DIR']",
                "pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin')":
                    "pretrained=(None if os.environ.get('ENCODER_SKIP_CLIP') == '1' else os.path.join(clip_path, 'open_clip_pytorch_model.bin'))",
                'nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()':
                    'nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32))',
                'self.learnable_relation[:, :].unsqueeze(0).cuda()':
                    'self.learnable_relation[:, :].unsqueeze(0).to(V.device)',
                'self.tokenizer(txt).cuda()': 'self.tokenizer(txt).to(img_x.device)',
            }
            for old, new in replacements.items():
                if code.count(old) != 1:
                    raise RuntimeError(f'Unexpected upstream source: {old}')
                code = code.replace(old, new)
        (vendor / name).write_text(code)

    for name in ['captions/test.json', 'image_splits/split.train.json', 'image_splits/split.test.json']:
        download(f'{DATA_URL}/{name}', ROOT / 'CIRThan' / name)

    assets = [
        ('results/enhanced_best 123.pt',
         'https://github.com/Lymous/ENCODER-Research/releases/download/demo-seed123/enhanced_best.123.pt', CHECKPOINT_SHA),
        ('ENCODER_repo/open_clip_pytorch_model.bin',
         'https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin', CLIP_SHA),
    ]
    for relative, url, sha in assets:
        # The complete research checkpoint already includes all CLIP weights.
        if relative.startswith('ENCODER_repo/') and os.environ.get('ENCODER_SKIP_CLIP') == '1':
            continue
        destination = ROOT / relative
        if args.assets_from and not destination.exists():
            local = args.assets_from / relative
            if local.is_file() and digest(local) == sha:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, destination)
        download(url, destination, sha)
    print('Model dependencies ready. Download CIRThan image/ and sketch/ from the official Drive link in README.')
    print('Then run: python scripts/check_demo.py')


if __name__ == '__main__':
    main()
