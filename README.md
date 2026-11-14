# ENCODER Research

**Enhanced ENCODER for Composed Thangka Image Retrieval via Domain Adaptation and Hierarchical Text Alignment**

Molion Syahputra Mulya and Cendra Putra, Universitas Negeri Surabaya (UNESA).

Research article: [DOI: 10.67561/jaids.v2i1.4](https://doi.org/10.67561/jaids.v2i1.4)

**Live demo: [Try ENCODER Thangka on Streamlit](https://encoder-thangka-demo.streamlit.app/)**

Explore sketch-and-text image retrieval directly in your browser, with no local installation required. The demo uses the ENCODER + CDP + LAT + HNM checkpoint (seed 123). Initial loading may take a few minutes if the app is waking up.

![Research illustration](picture/1.jpeg)

## About the Research

This research studies Composed Image Retrieval (CIR) for Thangka imagery. Users provide a sketch and a text description, and the system retrieves relevant images. Evaluation uses CIRThan with three description levels: L1 (concise), L2 (intermediate), and L3 (detailed).

The research extends ENCODER with four components:

- **Cross-Domain Projection (CDP):** residual adapters that bridge the representation gap between sketches and Thangka images.
- **Level-Aware Tags (LAT):** markers that distinguish L1, L2, and L3 descriptions.
- **Hard Negative Mining (HNM):** learning from negative pairs that are difficult to distinguish.
- **UCDR Domain Prompts:** trainable visual prompts for domain adaptation within the CLIP backbone.

## Demo Model

The demo uses **ENCODER + CDP + LAT + HNM, seed 123**, with a best single-seed result of **58.95% Avg R@1**. This differs from the full variant, which also includes UCDR. The reported full-model result is 55.61% Avg R@1; the re-run baseline achieves 52.91%.

| Description level | R@1 (%) | R@3 (%) | R@5 (%) |
| --- | ---: | ---: | ---: |
| L1 | 56.80 | 78.76 | 84.01 |
| L2 | 58.95 | 79.24 | 85.68 |
| L3 | 61.10 | 80.91 | 86.16 |
| Average | **58.95** | **79.64** | **85.28** |

R@K is the percentage of queries whose target image appears among the top K results. These figures come from research records, not a new evaluation performed while packaging this repository.

## Repository Contents

- `demo/`: Streamlit application with Retrieval, About, and Ask AI pages.
- `demo/gallery_cache/`: embeddings and IDs for 419 demo gallery images.
- [Demo checkpoint (GitHub Release)](https://github.com/Lymous/ENCODER-Research/releases/tag/demo-seed123): download `enhanced_best.123.pt` and save it as `results/enhanced_best 123.pt`.
- `picture/1.jpeg`: the illustration displayed in this README.

Research notebooks, custom training losses, optimizers, and training loops are not included. `demo/inference.py` contains the adapter architecture and inference logic needed to use the checkpoint. Setup downloads the base ENCODER implementation from its authors' repository rather than redistributing it here.

The checkpoint is approximately 606 MiB and is distributed through GitHub Releases, separately from Git history. Create `results/` if needed when downloading manually.

Checkpoint SHA-256:

```text
a57b84ad54abcfc5923f1e42cfa36303843a30a72572e49f5f033ffe5c923878
```

## Running the Demo

### Streamlit Community Cloud

The cloud entry point is `cloud/app.py`. It downloads the checkpoint and upstream
modules automatically. It uses CPU-only PyTorch from `cloud/requirements.txt`.
The complete checkpoint includes CLIP weights, so the cloud version skips the
separate CLIP download. The original local entry point is still available below.

The cloud version keeps all **419 gallery embeddings** searchable. It downloads
only requested images and sketches from the dataset authors' public Google Drive,
using `cloud/media_index.json` (838 public file links, not dataset media).
Gallery images are resized for display after download; their cached retrieval
embeddings are unchanged. Sketches retain their original resolution for inference.
The initial gallery shows six previews to reduce startup downloads. Concurrent
inference calls are serialized to limit peak memory usage.
Sketch uploads are limited to 10 MB per file.

To deploy from your Streamlit account:

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
2. Select repository `Lymous/ENCODER-Research`, branch `main`.
3. Set the main file path to **`cloud/app.py`**.
4. In advanced settings, select **Python 3.11**.
5. Optionally add `GEMINI_API_KEY` and `GEMINI_MODEL` in Secrets for Ask AI.
6. Deploy. The first startup downloads about 606 MiB of model weights and may take several minutes.

Google Drive download quotas or connectivity can temporarily prevent media from
loading. Missing images show a retry warning rather than being replaced by invented
results. Assets are cached on the instance and may need downloading again after a
reboot. Actual hosting resource limits and availability must be checked during
deployment; successful local testing alone does not establish a live deployment.

Verified locally in a fresh Python 3.11 environment installed from
`cloud/requirements.txt`: dependency checks, cloud navigation, public Drive media
downloads, and a real CPU retrieval query passed. Peak process RSS for that test
was 1,691,656 KiB (about 1.61 GiB), measured with `/usr/bin/time -v`. This is a
single-session local measurement, not a guarantee of hosting memory use or
multi-user capacity. Model weights already present locally were reused in the test.

To test the cloud entry point locally in a separate Python 3.11 environment:

```bash
python -m pip install -r cloud/requirements.txt
python -m streamlit run cloud/app.py
```

### Local Installation

Use **Python 3.11** in a new environment to avoid changing an existing research environment. CPU inference is supported; an NVIDIA GPU with CUDA can accelerate inference. For Linux/WSL:

```bash
git clone https://github.com/Lymous/ENCODER-Research.git
cd ENCODER-Research
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python scripts/setup_demo.py
```

For CPU-only PyTorch, replace `cu118` with `cpu` in the installation URL. On Windows, use `py -3.11 -m venv .venv` and `.venv\Scripts\activate`.

Setup downloads two base ENCODER modules, CIRThan metadata, CLIP weights, and the release checkpoint. Weight downloads total approximately 1.24 GB, excluding the dataset. Large files are SHA-256 verified. Setup can be rerun and skips valid files.

**Download images and sketches from the official CIRThan Google Drive link below**, then extract them into `CIRThan/image/` and `CIRThan/sketch/`. These media folders are not redistributed here. Then run:

```bash
python scripts/check_demo.py --inference
python -m streamlit run 'demo/🔍_Demo_Retrieval.py'
```

Open the local URL printed by Streamlit. Select a test example and description level, then click Search. The checker validates all gallery images and test sketches and runs a real query with `--inference`. Use `--device cpu` to force CPU testing.

For an existing dataset, set `CIRTHAN_DIR` before running the checker and Streamlit, for example `export CIRTHAN_DIR=/path/to/CIRThan`. To reuse existing local weights, run `python scripts/setup_demo.py --assets-from /path/to/project`. Setup copies weights with matching checksums without modifying the source project.

### Download Sources

| Component | Official source / download | Expected location |
| --- | --- | --- |
| ENCODER implementation | [Official AAAI25-ENCODER repository](https://github.com/iLearn-Lab/AAAI25-ENCODER); the former `Lee-zixu/ENCODER` URL redirects here. Setup downloads the required modules automatically. | `ENCODER_repo_new/`, including `model_try2.py` and `token_wise_matching.py` |
| CLIP ViT-B-32, LAION-2B weights | [Official LAION model](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) / [download open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin?download=true) | `ENCODER_repo/open_clip_pytorch_model.bin` |
| CIRThan metadata | [Official CIRThan repository](https://github.com/jinyuxu-whut/CIRThan), including `captions/` and `image_splits/` | `CIRThan/captions/` and `CIRThan/image_splits/` |
| CIRThan images and sketches | [Google Drive linked by the dataset authors](https://drive.google.com/drive/folders/1M7_l2DiWpwqhlvGpP5EuVEY0eM7-nC3a?usp=sharing) | Extract into `CIRThan/image/` and `CIRThan/sketch/` to match the metadata |

Setup pins ENCODER to commit `29a2a31d6a56f677bf450c3be7cdaef423fb7018` and CIRThan metadata to `b1499eb9393b12f1499a983f8fedfe7d14869024`. Both ENCODER modules are checksum-verified before local adjustments for absolute weight paths and CPU/CUDA device selection. Downloaded code is ignored by Git. The seed 123 checkpoint comes from [this research release](https://github.com/Lymous/ENCODER-Research/releases/tag/demo-seed123).

### Reproducibility Limits

Demo inference applies the sketch adapter and `[L1]`/`[L2]`/`[L3]` tags. The checkpoint does not contain `backbone.binding_decoder.learnable_relation`, because of how the upstream constructor moved that tensor to CUDA. The demo initializes it deterministically with seed 123 and rejects other weight mismatches. Successful demo execution does not guarantee reproduction of the reported 58.95% evaluation result. Gallery embeddings are loaded from the supplied cache rather than recomputed at startup.

Local verification: setup downloaded upstream code and metadata; weight checksums passed; all 419 gallery images and 419 test sketches were available; one CPU query returned its target at rank 1; and interface tests passed for retrieval (CUDA), About, and Ask AI without an API key. These initial tests used the research Python environment and existing dataset media and weights. A subsequent fresh CPU-only environment test is documented in the Cloud section above. Live Gemini responses have not been tested.

### Environment Installation Sources

Install the environment through a package manager. Refer to the [ENCODER installation instructions and requirements](https://github.com/iLearn-Lab/AAAI25-ENCODER) for the base model and these official sources for demo components:

- [PyTorch and CUDA options](https://pytorch.org/get-started/locally/).
- [Streamlit installation](https://docs.streamlit.io/get-started/installation).
- [Google Gen AI Python SDK](https://github.com/googleapis/python-genai), package `google-genai`, for Ask AI.

Demo dependency versions are listed in `requirements.txt`. For Ask AI, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and provide your own key. Run Streamlit from the repository root. Retrieval does not require an API key; Ask AI requires internet access, an API key, and a Gemini model available to your account.

## Attribution

This research builds on ENCODER and uses CLIP and CIRThan. Third-party code, base models, and datasets remain subject to their original ownership and licenses. See the article linked by DOI for the methodology, evaluation, and research references.
