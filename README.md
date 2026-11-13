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

Notebook training dan folder implementasi model tidak disertakan. File aplikasi demo tetap berisi kode integrasi dan wrapper `CIRThanEncoder` yang diperlukan oleh aplikasi asli. Checkpoint merupakan bobot model, bukan program yang dapat dijalankan sendiri.

Checkpoint berukuran sekitar 606 MiB dan didistribusikan melalui GitHub Release, terpisah dari riwayat Git. Buat folder `results/` bila belum ada setelah clone.

SHA-256 checkpoint:

```text
a57b84ad54abcfc5923f1e42cfa36303843a30a72572e49f5f033ffe5c923878
```

## Menjalankan pada lingkungan penelitian lokal

Repositori ini belum merupakan paket mandiri. Selain isi repositori, aplikasi membutuhkan:

| Komponen | Sumber resmi / unduhan | Lokasi yang diharapkan demo |
| --- | --- | --- |
| Implementasi ENCODER | [Repo resmi AAAI25-ENCODER](https://github.com/iLearn-Lab/AAAI25-ENCODER) (tautan lama `Lee-zixu/ENCODER` dialihkan ke sini). Gunakan tombol **Code > Download ZIP** atau clone repo. | `ENCODER_repo_new/`, termasuk `model_try2.py` dan `token_wise_matching.py` |
| Weight CLIP ViT-B-32, LAION-2B | [Model resmi LAION](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) / [unduh open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin?download=true) | `ENCODER_repo/open_clip_pytorch_model.bin` |
| Metadata CIRThan | [Repo resmi CIRThan](https://github.com/jinyuxu-whut/CIRThan), termasuk folder `captions/` dan `image_splits/` | `CIRThan/captions/` dan `CIRThan/image_splits/` |
| Gambar dan sketsa CIRThan | [Google Drive yang ditautkan oleh penulis dataset](https://drive.google.com/drive/folders/1M7_l2DiWpwqhlvGpP5EuVEY0eM7-nC3a?usp=sharing) | Ekstrak menjadi `CIRThan/image/` dan `CIRThan/sketch/` sesuai metadata |

`ENCODER_repo_new` adalah nama folder lokal proyek ini, bukan nama repositori resmi. Repo acuan menyediakan implementasi dasar ENCODER; demo mendefinisikan wrapper `CIRThanEncoder` dan menulis `model_enhanced.py` saat memuat model. Kompatibilitas versi upstream terbaru dengan checkpoint dan lingkungan lokal belum diuji ulang. Checkpoint CIRThan seed 123 tetap diambil dari [Release penelitian ini](https://github.com/Lymous/ENCODER-Research/releases/tag/demo-seed123), bukan checkpoint dataset lain milik ENCODER asli.

### Sumber instalasi environment

Environment dipasang melalui package manager, bukan diunduh sebagai bagian dataset. Gunakan [petunjuk instalasi dan requirements dari ENCODER](https://github.com/iLearn-Lab/AAAI25-ENCODER) sebagai acuan model, serta sumber resmi berikut untuk komponen demo:

- [PyTorch dan pilihan CUDA](https://pytorch.org/get-started/locally/). Implementasi model lokal saat ini menggunakan CUDA.
- [Instalasi Streamlit](https://docs.streamlit.io/get-started/installation).
- [Google Gen AI Python SDK](https://github.com/googleapis/python-genai), paket `google-genai`, untuk halaman Tanya AI.

Environment juga membutuhkan OpenCLIP, Transformers, NumPy, pandas, Pillow, dan einops. Requirements penelitian acuan tidak otomatis mencakup seluruh kebutuhan demo ini; versi pustaka perlu disesuaikan dengan lingkungan penelitian yang digunakan. Tautan di atas adalah sumber instalasi, bukan klaim bahwa semua versi terbaru telah diuji dengan checkpoint ini.

Tempatkan komponen tersebut pada root proyek dengan susunan yang sama seperti lingkungan penelitian asli. Backbone juga mencari `./open_clip_pytorch_model.bin`; pada Linux, buat symlink dari root proyek:

```bash
ln -s ENCODER_repo/open_clip_pytorch_model.bin open_clip_pytorch_model.bin
conda run -n LLMproject streamlit run 'demo/🔍_Demo_Retrieval.py'
```

Jalankan perintah dari root proyek. Untuk halaman Tanya AI, sediakan `GEMINI_API_KEY` melalui konfigurasi secrets Streamlit lokal. API key tidak disertakan dalam repositori; fitur ini memerlukan akses internet dan layanan Gemini.

## Atribusi

Penelitian dikembangkan di atas ENCODER dan menggunakan CLIP serta benchmark CIRThan. Kepemilikan kode, model dasar, dan dataset pihak ketiga tetap mengikuti sumber serta lisensinya masing-masing. Lihat artikel pada tautan DOI untuk konteks metode, evaluasi, dan referensi penelitian.
