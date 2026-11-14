"""Fetch only requested media from the dataset authors' public Drive folder."""
import json
import os
from pathlib import Path
import threading

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()


def ensure_media(path, dataset_dir):
    path = Path(path)
    if path.is_file() or os.environ.get('ENCODER_CLOUD') != '1':
        return str(path)
    import gdown
    relative = path.resolve().relative_to(Path(dataset_dir).resolve()).as_posix()
    index = json.loads((ROOT / 'cloud/media_index.json').read_text())
    if relative not in index:
        raise FileNotFoundError(f'No official download link for {relative}')
    with _LOCK:
        if path.is_file():
            return str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.download')
        try:
            result = gdown.download(id=index[relative], output=str(temporary), quiet=True,
                                    use_cookies=False, timeout=60)
            if not result:
                raise RuntimeError('Dataset download failed. Please retry later.')
            with Image.open(temporary) as image:
                image.verify()
            if relative.startswith('image/'):
                # Gallery images are display-only; retrieval uses cached embeddings.
                with Image.open(temporary) as image:
                    thumbnail = image.convert('RGB')
                    thumbnail.thumbnail((768, 768))
                    thumbnail.save(temporary, format='PNG')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return str(path)
