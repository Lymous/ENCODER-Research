"""Index public Drive links for test media, without redistributing the dataset."""
import json
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = ROOT / 'CIRThan'
    paths = {}
    for split in ('train', 'test'):
        paths.update(json.loads((metadata / 'image_splits' / f'split.{split}.json').read_text()))
    ids = json.loads((ROOT / 'demo/gallery_cache/gallery_ids.json').read_text())
    samples = json.loads((metadata / 'captions/test.json').read_text())
    wanted = {paths[i].removeprefix('./') for i in ids}
    wanted.update('sketch/' + s['sketch_id'].removeprefix('sketch') + '.png' for s in samples)
    files = gdown.download_folder(id='1M7_l2DiWpwqhlvGpP5EuVEY0eM7-nC3a',
                                 skip_download=True, quiet=True, timeout=60)
    index = {f.path: f.id for f in files if f.path in wanted}
    if set(index) != wanted:
        raise RuntimeError(f'Missing Drive paths: {wanted - set(index)}')
    (ROOT / 'cloud/media_index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    print(f'Indexed {len(index)} public media links. No media files downloaded.')


if __name__ == '__main__':
    main()
