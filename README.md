# ENCODER Research

**Enhanced ENCODER for Composed Thangka Image Retrieval via Domain Adaptation and Hierarchical Text Alignment**

Molion Syahputra Mulya dan Cendra Putra, Universitas Negeri Surabaya (UNESA).

Artikel penelitian: [DOI: 10.67561/jaids.v2i1.4](https://doi.org/10.67561/jaids.v2i1.4)

![Ilustrasi penelitian](picture/1.jpeg)

## Tentang penelitian

Penelitian ini membahas Composed Image Retrieval (CIR) untuk gambar Thangka: pengguna memberikan sketsa dan deskripsi teks, kemudian sistem mencari gambar yang paling relevan. Evaluasi menggunakan benchmark CIRThan dengan tiga tingkat deskripsi: L1 (ringkas), L2 (sedang), dan L3 (detail).

Penelitian mengembangkan ENCODER dengan empat komponen:

- **Cross-Domain Projection (CDP):** adapter residual untuk menjembatani perbedaan representasi sketsa dan gambar Thangka.
- **Level-Aware Tags (LAT):** penanda tingkat deskripsi untuk membantu model membedakan informasi teks L1, L2, dan L3.
- **Hard Negative Mining (HNM):** pembelajaran dari pasangan negatif yang sulit dibedakan.
- **UCDR Domain Prompts:** prompt visual yang dapat dilatih untuk adaptasi domain pada backbone CLIP.

## Model yang digunakan demo

Demo menggunakan checkpoint **ENCODER + CDP + LAT + HNM, seed 123**, dengan hasil terbaik satu seed **Avg R@1 = 58,95%**. Checkpoint ini berbeda dari varian penuh yang juga menggunakan UCDR. Hasil varian penuh yang dilaporkan dalam penelitian adalah Avg R@1 55,61%; baseline re-run adalah 52,91%.

| Tingkat deskripsi | R@1 (%) | R@3 (%) | R@5 (%) |
| --- | ---: | ---: | ---: |
| L1 | 56,80 | 78,76 | 84,01 |
| L2 | 58,95 | 79,24 | 85,68 |
| L3 | 61,10 | 80,91 | 86,16 |
| Rata-rata | **58,95** | **79,64** | **85,28** |

R@K adalah persentase kueri yang gambar targetnya ditemukan dalam K hasil teratas. Angka di atas berasal dari catatan hasil penelitian lokal, bukan evaluasi ulang pada saat pengemasan repositori.

## Isi repositori

- `demo/`: aplikasi Streamlit dengan halaman retrieval, informasi penelitian, dan Tanya AI.
- `demo/gallery_cache/`: embedding dan ID untuk 419 gambar galeri demo.
- [Checkpoint demo (GitHub Release)](https://github.com/Lymous/ENCODER-Research/releases/tag/demo-seed123): unduh `enhanced_best.123.pt`, kemudian simpan sebagai `results/enhanced_best 123.pt`.
- `picture/1.jpeg`: gambar yang ditampilkan pada README.

Notebook, loss kustom, optimizer, dan loop training penelitian tidak disertakan. `demo/inference.py` hanya memuat arsitektur adapter dan inferensi untuk menggunakan checkpoint. Kode dasar ENCODER diunduh dari repo penulis melalui `scripts/setup_demo.py`, bukan diunggah ulang. Checkpoint merupakan bobot model, bukan program yang dapat dijalankan sendiri.

Checkpoint berukuran sekitar 606 MiB dan didistribusikan melalui GitHub Release, terpisah dari riwayat Git. Buat folder `results/` bila belum ada setelah clone.

SHA-256 checkpoint:

```text
a57b84ad54abcfc5923f1e42cfa36303843a30a72572e49f5f033ffe5c923878
```

## Menjalankan demo

Gunakan **Python 3.11** dalam environment baru agar instalasi tidak mengubah environment penelitian yang sudah ada. CPU didukung; GPU NVIDIA dengan CUDA dapat mempercepat inferensi. Perintah berikut untuk Linux/WSL:

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

Untuk instalasi PyTorch CPU-only, ganti `cu118` dengan `cpu` pada URL perintah PyTorch. Di Windows gunakan `py -3.11 -m venv .venv` dan `.venv\Scripts\activate`.

Setup mengunduh dua modul dasar ENCODER, metadata CIRThan, weight CLIP, dan checkpoint dari Release. Total unduhan bobot sekitar 1,24 GB, belum termasuk dataset. File besar diverifikasi SHA-256. Setup dapat dijalankan ulang dan melewati file yang sudah valid.

**Gambar dan sketsa perlu diunduh dari Google Drive resmi CIRThan pada tabel di bawah**, lalu ekstrak sehingga `CIRThan/image/` dan `CIRThan/sketch/` berada langsung di root dataset. Kedua folder media tidak diunggah ulang di repo ini. Setelah itu:

```bash
python scripts/check_demo.py --inference
python -m streamlit run 'demo/🔍_Demo_Retrieval.py'
```

Buka URL lokal yang dicetak Streamlit. Pilih contoh test set, tingkat deskripsi, lalu klik Cari. `check_demo.py` memeriksa seluruh gambar galeri dan sketsa test, serta menjalankan satu kueri nyata dengan `--inference`. Gunakan `--device cpu` untuk memaksa tes CPU.

Jika dataset sudah tersedia di tempat lain, atur `CIRTHAN_DIR` ke folder tersebut sebelum menjalankan pemeriksaan dan Streamlit. Contoh Linux: `export CIRTHAN_DIR=/path/ke/CIRThan`. Untuk memakai bobot yang sudah tersedia di proyek lokal, jalankan `python scripts/setup_demo.py --assets-from /path/ke/proyek`. Setup menyalin bobot yang sesuai checksum dan tidak mengubah proyek sumber.

### Sumber unduhan

| Komponen | Sumber resmi / unduhan | Lokasi yang diharapkan demo |
| --- | --- | --- |
| Implementasi ENCODER | [Repo resmi AAAI25-ENCODER](https://github.com/iLearn-Lab/AAAI25-ENCODER) (tautan lama `Lee-zixu/ENCODER` dialihkan ke sini). Gunakan tombol **Code > Download ZIP** atau clone repo. | `ENCODER_repo_new/`, termasuk `model_try2.py` dan `token_wise_matching.py` |
| Weight CLIP ViT-B-32, LAION-2B | [Model resmi LAION](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) / [unduh open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin?download=true) | `ENCODER_repo/open_clip_pytorch_model.bin` |
| Metadata CIRThan | [Repo resmi CIRThan](https://github.com/jinyuxu-whut/CIRThan), termasuk folder `captions/` dan `image_splits/` | `CIRThan/captions/` dan `CIRThan/image_splits/` |
| Gambar dan sketsa CIRThan | [Google Drive yang ditautkan oleh penulis dataset](https://drive.google.com/drive/folders/1M7_l2DiWpwqhlvGpP5EuVEY0eM7-nC3a?usp=sharing) | Ekstrak menjadi `CIRThan/image/` dan `CIRThan/sketch/` sesuai metadata |

Setup menggunakan commit ENCODER `29a2a31d6a56f677bf450c3be7cdaef423fb7018` dan metadata CIRThan `b1499eb9393b12f1499a983f8fedfe7d14869024`. Dua modul ENCODER diverifikasi checksum sebelum penyesuaian lokal untuk path weight absolut dan pemilihan device CPU/CUDA. Kode yang diunduh diabaikan oleh Git. Checkpoint CIRThan seed 123 tetap diambil dari [Release penelitian ini](https://github.com/Lymous/ENCODER-Research/releases/tag/demo-seed123).

### Batas reproduksi hasil

Inferensi demo menerapkan adapter sketsa dan tag `[L1]`/`[L2]`/`[L3]`. Checkpoint yang tersedia tidak menyimpan tensor `backbone.binding_decoder.learnable_relation`, akibat cara konstruktor upstream memindahkan tensor ke CUDA. Demo menginisialisasi tensor tersebut secara deterministik dengan seed 123 dan menolak ketidakcocokan bobot lain. Karena itu, keberhasilan demo bukan jaminan reproduksi angka evaluasi 58,95%; angka tersebut adalah hasil penelitian yang dilaporkan. Cache galeri menggunakan embedding yang sudah disediakan, bukan dihitung ulang saat startup.

Verifikasi lokal: setup mengunduh kode acuan dan metadata, checksum bobot lolos, seluruh 419 gambar galeri dan 419 sketsa test tersedia, satu kueri CPU menghasilkan target di rank 1, dan tes antarmuka retrieval (CUDA), Tentang, serta Tanya AI tanpa key lulus. Pengujian menggunakan environment Python penelitian yang sudah tersedia; instalasi environment baru dan respons API Gemini belum diuji. Media dataset dan bobot yang sudah ada digunakan ulang saat pengujian.

### Sumber instalasi environment

Environment dipasang melalui package manager, bukan diunduh sebagai bagian dataset. Gunakan [petunjuk instalasi dan requirements dari ENCODER](https://github.com/iLearn-Lab/AAAI25-ENCODER) sebagai acuan model, serta sumber resmi berikut untuk komponen demo:

- [PyTorch dan pilihan CUDA](https://pytorch.org/get-started/locally/).
- [Instalasi Streamlit](https://docs.streamlit.io/get-started/installation).
- [Google Gen AI Python SDK](https://github.com/googleapis/python-genai), paket `google-genai`, untuk halaman Tanya AI.

Versi dependensi demo dicatat dalam `requirements.txt`. Untuk halaman Tanya AI, salin `.streamlit/secrets.toml.example` menjadi `.streamlit/secrets.toml` dan isi key sendiri. Jalankan Streamlit dari root repo. Retrieval tidak membutuhkan API key; Tanya AI membutuhkan internet, API key, dan model Gemini yang tersedia pada akun pengguna.

## Atribusi

Penelitian dikembangkan di atas ENCODER dan menggunakan CLIP serta benchmark CIRThan. Kepemilikan kode, model dasar, dan dataset pihak ketiga tetap mengikuti sumber serta lisensinya masing-masing. Lihat artikel pada tautan DOI untuk konteks metode, evaluasi, dan referensi penelitian.
