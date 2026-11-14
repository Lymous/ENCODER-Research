"""Inference-only checkpoint adapter. No training loop, loss or optimizer."""
import os
from pathlib import Path
import sys
import warnings
import threading

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(os.environ.get('CIRTHAN_DIR', ROOT / 'CIRThan')).resolve()
_INFERENCE_LOCK = threading.Lock()


def load_model(device_name=None):
    import open_clip
    vendor = ROOT / 'ENCODER_repo_new'
    if not (vendor / 'model_try2.py').is_file():
        raise FileNotFoundError('Run python scripts/setup_demo.py first.')
    sys.path.insert(0, str(vendor))
    os.environ['ENCODER_CLIP_DIR'] = str(ROOT / 'ENCODER_repo')
    from model_try2 import Encoder

    class CIRThanEncoder(Encoder):
        def __init__(self):
            super().__init__(512, 0.5, 1, 0.1, 2, 1, True)
            self.sketch_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(512), torch.nn.Linear(512, 512),
                torch.nn.GELU(), torch.nn.Linear(512, 512))
            self.thangka_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(512), torch.nn.Linear(512, 512),
                torch.nn.GELU(), torch.nn.Linear(512, 512))

        def compose_feature(self, ref, mod):
            ref_tokens, text_tokens, similarity, relation, entity, action = (
                self.backbone.extract_img_fea_patch_selection(ref, mod))
            ref_tokens = ref_tokens + self.sketch_proj(ref_tokens)
            return self.affine(ref_tokens, text_tokens), similarity, relation, entity, action

        def target_fea(self, tag):
            tokens = self.backbone.extract_img_fea(tag)
            return tokens + self.thangka_proj(tokens)

    device = torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    # The upstream CUDA constructor did not register this relation tensor in
    # state_dict. Use a fixed initialization for repeatable demo queries.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = CIRThanEncoder()
    state = torch.load(ROOT / 'results' / 'enhanced_best 123.pt', map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {'backbone.binding_decoder.learnable_relation'}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f'Incompatible checkpoint: missing={missing}, unexpected={unexpected}')
    if missing:
        warnings.warn('Checkpoint omits learnable_relation; demo initializes it with seed 123. '
                      'This demo does not reproduce the complete published evaluation.')
    del state
    model = model.to(device).eval()
    from open_clip.transform import image_transform
    preprocess = image_transform(model.backbone.clip.visual.image_size, is_train=False)
    return model, preprocess, device


def retrieve(image, text, top_k, model, preprocess, device, gallery, ids, level=1):
    if level not in (1, 2, 3):
        raise ValueError('Level must be 1, 2 or 3.')
    query = f'[L{level}] {text.strip()}'
    with _INFERENCE_LOCK, torch.inference_mode():
        embedding = model.extract_retrieval_compose(
            preprocess(image).unsqueeze(0).to(device), [query])
    scores = (embedding.cpu().numpy() @ gallery.T).squeeze(0)
    if not np.isfinite(scores).all():
        raise RuntimeError('Non-finite retrieval scores.')
    indices = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in indices]
