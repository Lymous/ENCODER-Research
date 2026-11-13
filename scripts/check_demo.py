"""Validate assets and optionally run one real retrieval query."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'demo'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    from inference import DATASET_DIR, load_model, retrieve
    cache = ROOT / 'demo' / 'gallery_cache'
    ids = json.loads((cache / 'gallery_ids.json').read_text())
    gallery = np.load(cache / 'gallery_embeddings.npy')
    assert gallery.shape == (len(ids), 512) and np.isfinite(gallery).all()
    paths = {}
    for split in ('train', 'test'):
        paths.update(json.loads((DATASET_DIR / 'image_splits' / f'split.{split}.json').read_text()))
    samples = json.loads((DATASET_DIR / 'captions' / 'test.json').read_text())
    required = [ROOT / 'ENCODER_repo_new' / 'model_try2.py',
                ROOT / 'ENCODER_repo_new' / 'token_wise_matching.py',
                ROOT / 'ENCODER_repo' / 'open_clip_pytorch_model.bin',
                ROOT / 'results' / 'enhanced_best 123.pt']
    required += [DATASET_DIR / paths[i].removeprefix('./') for i in ids]
    required += [DATASET_DIR / 'sketch' / (s['sketch_id'].removeprefix('sketch') + '.png') for s in samples]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit('Missing assets (download/extract using README):\n' + '\n'.join(missing[:10]))
    print(f'Assets OK: {len(ids)} gallery images, {len(samples)} test samples.')
    if args.inference:
        model, preprocess, device = load_model(args.device)
        sample = samples[0]
        sketch = DATASET_DIR / 'sketch' / (sample['sketch_id'].removeprefix('sketch') + '.png')
        with Image.open(sketch) as image:
            results = retrieve(image.convert('RGB'), sample['level1'], 5, model, preprocess, device, gallery, ids)
        assert len(results) == 5
        print(f'Inference OK ({device}), ground truth: {sample["thangka_id"]}')
        for rank, (image_id, score) in enumerate(results, 1):
            print(f'{rank}: {image_id} {score:.6f}')


if __name__ == '__main__':
    main()
