import streamlit as st

st.set_page_config(page_title="Tentang - CIR Thangka", page_icon="📖", layout="wide")

st.title("📖 Tentang Penelitian Ini")
st.caption("Composed Image Retrieval untuk Gambar Thangka menggunakan Model ENCODER")

st.divider()

# ── 1. Apa itu CIR ────────────────────────────────────────────────────────────
st.header("🔍 Apa itu Composed Image Retrieval (CIR)?")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
Composed Image Retrieval (CIR) adalah tugas pencarian gambar yang menggunakan
**dua jenis input sekaligus**, yaitu sebuah **gambar referensi** (sketsa) dan sebuah
**deskripsi teks**. Model menggabungkan keduanya untuk menemukan gambar target
yang paling relevan dari sebuah galeri.

Contoh: *"Cari gambar Thangka yang serupa dengan sketsa ini, tetapi dengan figur
Buddha di tengah dan latar lotus."*

Ini berbeda dari pencarian gambar biasa yang hanya menggunakan gambar atau teks
saja, karena CIR memungkinkan pencarian yang jauh lebih spesifik dan ekspresif.
""")
with col2:
    st.info(
        "**Sketsa** (gambar referensi)\n\n"
        "+ **Teks** (deskripsi modifikasi)\n\n"
        "↓\n\n"
        "**Gambar Thangka** yang relevan",
    )

st.divider()

# ── 2. Dataset CIRThan ────────────────────────────────────────────────────────
st.header("🖼️ Dataset CIRThan & Gambar Thangka")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Thangka** adalah lukisan keagamaan Buddha Tibet yang memiliki kompleksitas
visual dan simbolik sangat tinggi, di mana setiap figur, warna, dan atribut memiliki
makna ikonografis tersendiri.

**CIRThan** adalah benchmark CIR pertama untuk domain Thangka, terdiri dari:
- **1.868 gambar** Thangka sebagai galeri
- **Sketsa** garis yang dipasangkan dengan gambar Thangka
- **3 level deskripsi teks** per pasang (L1, L2, L3)
""")
with col2:
    st.markdown("""
**Tiga Level Deskripsi Teks:**

| Level | Keterangan | Contoh panjang |
|---|---|---|
| **L1** | Ringkas | ~10 kata |
| **L2** | Sedang | ~30 kata |
| **L3** | Detail | ~50 kata |

Semakin tinggi levelnya, deskripsi semakin spesifik mengenai atribut ikonografis
figur dalam gambar Thangka.
""")

st.divider()

# ── 3. Model ──────────────────────────────────────────────────────────────────
st.header("⚙️ Model yang Digunakan")
st.markdown("""
Penelitian ini memodifikasi model **ENCODER** (Li et al., 2025) dengan menambahkan
empat komponen baru untuk mengatasi keterbatasannya pada domain Thangka:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**1. CDP (Cross-Domain Projection)**
Menjembatani perbedaan karakteristik visual antara sketsa dan gambar Thangka.
Dua adapter residual ditambahkan setelah encoder CLIP, yaitu satu untuk modalitas
sketsa dan satu untuk modalitas Thangka, tanpa mengubah bobot backbone CLIP.
*(~1,05 juta parameter tambahan)*

**2. LAT (Level-Aware Tags)**
Menambahkan token penanda level (L1/L2/L3) pada embedding teks sehingga model
memproses setiap level deskripsi secara berbeda. Tanpa LAT, embedding dari
ketiga level tumpang tindih dalam ruang teks bersama.
""")
with col2:
    st.markdown("""
**3. HNM (Hard Negative Mining)**
Menambahkan triplet loss yang secara eksplisit menargetkan *hard negative*
terdekat dalam ruang embedding, sehingga model belajar dari pasangan yang
paling sulit dibedakan. Komponen ini berkontribusi pada stabilitas hasil
lintas variasi random seed.

**4. UCDR Domain Prompts**
Token prompt yang dapat dilatih untuk mengadaptasi encoder visual CLIP agar
lebih sensitif terhadap karakteristik visual khas gambar Thangka, mengikuti
pendekatan Visual Prompt Tuning.
""")

st.divider()

# ── 4. Hasil ──────────────────────────────────────────────────────────────────
st.header("📊 Hasil Evaluasi pada CIRThan")
st.markdown("Metrik: **Recall@K (R@K, %)**, yaitu persentase kueri yang ground truth-nya masuk dalam Top-K hasil retrieval.")

col_tbl, col_gap, col_highlight = st.columns([3, 0.3, 1.5])
with col_tbl:
    import pandas as pd
    df = pd.DataFrame({
        "Model": [
            "COMBINER (Baldrati et al.)",
            "CLIP4CIR (Baldrati et al.)",
            "BiLLIP4CIR (Liu et al.)",
            "CALA (Jiang et al.)",
            "ENCODER sebagai baseline (Li et al.)",
            "ENCODER re-run kami",
            "✨ ENCODER+CDP+LAT+HNM (kami)",
        ],
        "Avg R@1": ["≤30%", "≤30%", "≤30%", "≤30%", "52,03%", "52,91%", "58,95%"],
        "Avg R@3": ["n/a", "n/a", "n/a", "n/a", "75,50%", "74,30%", "79,64%"],
        "Avg R@5": ["n/a", "n/a", "n/a", "n/a", "84,17%", "81,07%", "85,28%"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Semua metode dievaluasi pada himpunan uji CIRThan. Avg merupakan rata-rata makro atas L1, L2, dan L3.")

with col_highlight:
    st.metric("Avg R@1 (Best)", "58,95%", delta="+6,92 pp vs baseline")
    st.metric("Avg R@3 (Best)", "79,64%", delta="+4,14 pp vs baseline")
    st.metric("Avg R@5 (Best)", "85,28%", delta="+1,11 pp vs baseline")

st.divider()

# ── 5. Per-level breakdown ────────────────────────────────────────────────────
st.header("📈 Detail Per-Level (Model Terbaik)")
st.markdown("Model terbaik: **ENCODER+CDP+LAT+HNM**, seed 123.")

df_level = pd.DataFrame({
    "Level": ["L1 (Ringkas)", "L2 (Sedang)", "L3 (Detail)", "Rata-rata"],
    "R@1 (%)": ["56,80", "58,95", "61,10", "58,95"],
    "R@3 (%)": ["78,76", "79,24", "80,91", "79,64"],
    "R@5 (%)": ["84,01", "85,68", "86,16", "85,28"],
})
st.dataframe(df_level, use_container_width=True, hide_index=True)
st.caption(
    "Deskripsi yang lebih detail (L3) konsisten menghasilkan R@K yang lebih tinggi, "
    "yang menunjukkan efektivitas LAT dalam memanfaatkan struktur hierarkis teks."
)

st.divider()
st.markdown(
    "_Penelitian ini merupakan bagian dari skripsi S1 Program Studi Sistem Informasi, "
    "Universitas Negeri Surabaya (UNESA). Penulis: Molion Syahputra Mulya._"
)
