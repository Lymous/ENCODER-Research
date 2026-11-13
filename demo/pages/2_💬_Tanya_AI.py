import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Tanya AI", page_icon="💬", layout="wide")

# ── System prompt konteks penelitian ─────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah asisten AI yang membantu pengunjung pameran memahami penelitian artikel ilmiah berikut.

=== INFORMASI ARTIKEL ===
Judul: Enhanced ENCODER for Composed Thangka Image Retrieval via Domain Adaptation and Hierarchical Text Alignment
Penulis: Molion Syahputra Mulya, Cendra Putra
Institusi: Program Studi Sistem Informasi, Fakultas Teknik, Universitas Negeri Surabaya (UNESA), Indonesia
DOI: https://doi.org/10.67561/jaids.v2i1.4

=== ABSTRAK ARTIKEL (ASLI) ===
Composed image retrieval (CIR) for Thangka imagery presents a challenging domain gap: queries combine a sketch with hierarchical natural language descriptions, yet general-purpose vision–language models struggle with the dense iconographic structures of this culturally specific domain. This paper proposes four parameter-efficient modifications to ENCODER, the strongest supervised baseline on the CIRThan benchmark: Cross-Domain Projection (CDP) residual adapters, Level-Aware Tags (LAT), Hard Negative Mining (HNM), and UCDR Domain Prompts. All modifications are integrated via component-level code transplantation and fine-tuned for 20 epochs on a Kaggle Tesla T4 GPU. Our re-run of ENCODER achieves 52.91% average R@1. UCDR domain prompts alone contribute the largest single gain (+1.43 pp), more than the other three modifications combined (+0.31 pp), confirming that visual domain adaptation is the primary bottleneck. The full model (UCDR+CDP+LAT+HNM) achieves 55.61% average R@1 (+2.70 pp over baseline), adding only 2.23M trainable parameters to the 151M frozen CLIP backbone. A full component-wise ablation across 16 variants shows that all top 5 combinations include UCDR domain prompts. Multi-seed evaluation across three random seeds further shows that ENCODER+CDP+LAT+HNM achieves the best single-seed result of 58.95% average R@1 (seed 123), while the full model maintains more consistent performance across seeds (53.14%–56.09%). The L3–L1 spread narrows from 4.06 pp (ENCODER) to 1.67 pp (full model), indicating improved handling of coarse-level queries.

=== DETAIL TEKNIS ===
Benchmark: CIRThan — dataset CIR pertama untuk domain Thangka (sketsa + teks hierarkis L1/L2/L3, 1.868 gambar)
Model dasar: ENCODER (Li et al., AAAI 2025), Avg R@1 = 52.03% (paper asli), 52.91% (re-run kami)

Empat modifikasi yang diusulkan:
1. CDP (Cross-Domain Projection): dua adapter residual setelah encoder CLIP yang dibekukan, satu untuk modalitas sketsa dan satu untuk modalitas Thangka, menjembatani kesenjangan distribusi visual antar domain (~1,05 juta parameter tambahan)
2. LAT (Level-Aware Tags): token penanda level [L1]/[L2]/[L3] yang digabungkan ke embedding teks sehingga model memproses setiap level deskripsi secara berbeda dan tidak tumpang tindih dalam ruang embedding bersama
3. HNM (Hard Negative Mining): triplet loss yang secara eksplisit menargetkan hard negative terdekat dalam ruang embedding, meningkatkan stabilitas hasil lintas variasi random seed
4. UCDR Domain Prompts: token prompt yang dapat dilatih mengikuti pendekatan Visual Prompt Tuning untuk mengadaptasi encoder CLIP terhadap statistik visual khas Thangka

Hasil per level (ENCODER+CDP+LAT+HNM, seed 123 — best single-seed):
- L1 (ringkas): R@1=56,80%, R@3=78,76%, R@5=84,01%
- L2 (sedang):  R@1=58,95%, R@3=79,24%, R@5=85,68%
- L3 (detail):  R@1=61,10%, R@3=80,91%, R@5=86,16%
- Rata-rata:     R@1=58,95%, R@3=79,64%, R@5=85,28%

Hasil model penuh UCDR+CDP+LAT+HNM (konsisten lintas seed):
- Rata-rata R@1: 55,61% (+2,70 pp vs baseline)
- Rentang lintas seed: 53,14%–56,09% (lebih stabil)

=== KONTEKS UMUM ===
- Thangka: lukisan keagamaan Buddha Tibet dengan struktur ikonografis yang sangat khas dan kompleks
- CIR: tugas pencarian gambar menggunakan gabungan gambar referensi (sketsa) dan deskripsi teks modifikasi
- Recall@K (R@K): persentase kueri yang gambar ground truth-nya masuk dalam Top-K hasil retrieval
- CLIP: model vision-language dari OpenAI yang digunakan sebagai backbone, dibekukan selama training
- Dataset galeri demo ini: 419 gambar Thangka dari himpunan uji CIRThan

Jawab dengan bahasa Indonesia yang sederhana dan ramah, cocok untuk pengunjung umum yang belum tentu berlatar belakang teknis. Jawaban maksimal 3-4 kalimat agar mudah dipahami. Jika ada pertanyaan di luar topik penelitian ini, arahkan kembali ke topik dengan sopan."""

# ── Pertanyaan siap klik ───────────────────────────────────────────────────────
PRESET_QUESTIONS = [
    "Apa itu gambar Thangka?",
    "Bagaimana sketsa dan teks digabungkan oleh model?",
    "Apa bedanya deskripsi L1, L2, dan L3?",
    "Mengapa model ini lebih baik dari metode lain?",
    "Apa fungsi CDP dalam model ini?",
    "Apa itu Recall@K dan mengapa digunakan?",
]

# ── Inisialisasi Gemini ───────────────────────────────────────────────────────
def get_gemini_response(user_message: str, history: list) -> str:
    import time
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
    except FileNotFoundError:
        return "API key belum diisi. Silakan isi GEMINI_API_KEY di file `.streamlit/secrets.toml`."
    if not api_key or api_key == "ISI_API_KEY_DISINI":
        return "API key belum diisi. Silakan isi GEMINI_API_KEY di file `.streamlit/secrets.toml`."
    client = genai.Client(api_key=api_key)
    contents = []
    for msg in history[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            return response.text
        except Exception as e:
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err:
                if attempt < 2:
                    time.sleep(3)
                    continue
                return "Server Gemini sedang sibuk. Coba lagi dalam beberapa detik."
            return f"Terjadi kesalahan: {err}"
    return "Server Gemini sedang sibuk. Coba lagi dalam beberapa detik."

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("💬 Tanya AI tentang Penelitian Ini")
st.caption("Didukung oleh Gemini · Tanyakan apa saja tentang CIR, Thangka, atau model yang digunakan")

# Session state untuk chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Pertanyaan siap klik ──────────────────────────────────────────────────────
st.markdown("**Pertanyaan yang sering ditanyakan:**")
cols = st.columns(3)
for i, q in enumerate(PRESET_QUESTIONS):
    if cols[i % 3].button(q, use_container_width=True, key=f"preset_{i}"):
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.spinner("Memproses..."):
            answer = get_gemini_response(q, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

st.divider()

# ── Riwayat chat ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input pertanyaan bebas ────────────────────────────────────────────────────
user_input = st.chat_input("Tulis pertanyaanmu di sini...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Memproses..."):
            answer = get_gemini_response(user_input, st.session_state.chat_history)
        st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ── Tombol reset ──────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    if st.button("🗑️ Hapus riwayat percakapan", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()
