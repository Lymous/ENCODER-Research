"""
CIR Thangka Demo — Streamlit app
Jalankan:
  streamlit run demo/🔍_Demo_Retrieval.py
"""

import os, sys, json, random
import numpy as np
import torch
import streamlit as st
import pandas as pd
from PIL import Image
from inference import DATASET_DIR as DATA_PATH, load_model as load_inference_model, retrieve

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR    = os.path.dirname(os.path.abspath(__file__))
ENCODER_DIR = os.path.join(ROOT, 'ENCODER_repo_new')
DATASET_DIR = str(DATA_PATH)
CHECKPOINT  = os.path.join(ROOT, 'results', 'enhanced_best 123.pt')
CACHE_DIR   = os.path.join(DEMO_DIR, 'gallery_cache')

sys.path.insert(0, ENCODER_DIR)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 Demo Retrieval",
    page_icon="🔍",
    layout="wide",
)

# ── Load gallery metadata ────────────────────────────────────────────────────
@st.cache_resource
def load_gallery_meta():
    with open(os.path.join(CACHE_DIR, 'gallery_ids.json')) as f:
        ids = json.load(f)
    embs = np.load(os.path.join(CACHE_DIR, 'gallery_embeddings.npy'))

    split_path = {}
    for fname in ('split.train.json', 'split.test.json'):
        with open(os.path.join(DATASET_DIR, 'image_splits', fname)) as f:
            split_path.update(json.load(f))

    id_to_path = {}
    for img_id in ids:
        rel = split_path.get(img_id, '')
        id_to_path[img_id] = os.path.join(DATASET_DIR, rel.lstrip('./'))

    return ids, embs, id_to_path


@st.cache_data
def load_test_samples():
    with open(os.path.join(DATASET_DIR, 'captions', 'test.json')) as f:
        return json.load(f)


# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return load_inference_model()


# ── Session state defaults ───────────────────────────────────────────────────
for key, default in [
    ('sketch_img',   None),
    ('gt_id',        None),
    ('sketch_label', ''),
    ('levels',       None),   # dict: {1: '...', 2: '...', 3: '...'}
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🖼️ Composed Image Retrieval — Thangka Demo")
st.caption("Model: ENCODER + CDP + LAT + HNM (seed 123) · Galeri: 419 gambar Thangka")

try:
    gallery_ids, gallery_embs, id_to_path = load_gallery_meta()
    test_samples = load_test_samples()
except FileNotFoundError:
    st.error("Aset demo belum tersedia. Jalankan python scripts/setup_demo.py dan ikuti langkah dataset di README.")
    st.stop()

with st.sidebar:
    st.header("Input Kueri")

    # ── Tombol Coba Contoh ──────────────────────────────────────────────────
    if st.button("🎲 Coba Contoh dari Test Set", use_container_width=True):
        sample = random.choice(test_samples)
        sketch_num  = sample['sketch_id'].replace('sketch', '')
        sketch_path = os.path.join(DATASET_DIR, 'sketch', f"{sketch_num}.png")
        if os.path.exists(sketch_path):
            st.session_state.sketch_img   = Image.open(sketch_path).convert('RGB')
            st.session_state.gt_id        = sample['thangka_id']
            st.session_state.sketch_label = sample['sketch_id']
            st.session_state.levels       = {
                1: sample['level1'],
                2: sample['level2'],
                3: sample['level3'],
            }
            st.session_state["text_input"] = sample['level1']
        else:
            st.warning(f"File sketsa tidak ditemukan: {sketch_path}")

    st.divider()

    # ── Upload manual ───────────────────────────────────────────────────────
    sketch_file = st.file_uploader("Atau upload sketsa sendiri", type=["png", "jpg", "jpeg"])
    if sketch_file:
        st.session_state.sketch_img   = Image.open(sketch_file).convert('RGB')
        st.session_state.gt_id        = None
        st.session_state.sketch_label = sketch_file.name
        st.session_state.levels       = None

    # ── Level selector (hanya muncul kalau dari Coba Contoh) ───────────────
    level_choice = 1
    if st.session_state.levels:
        st.markdown("**Level deskripsi teks**")
        level_choice = st.radio(
            "Level",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "L1 — Ringkas",
                2: "L2 — Sedang",
                3: "L3 — Detail",
            }[x],
            horizontal=True,
            key="level_radio",
            label_visibility="collapsed",
        )
        selected_text = st.session_state.levels[level_choice]
        # Sync ke text_input setiap kali level berubah
        if st.session_state.get("text_input") != selected_text:
            st.session_state["text_input"] = selected_text
            st.rerun()
    else:
        level_choice = st.selectbox("Level deskripsi", [1, 2, 3], format_func=lambda x: f"L{x}")

    text_query = st.text_area(
        "Deskripsi teks",
        placeholder="Contoh: a thangka with a central Buddha figure surrounded by lotus flowers",
        key="text_input",
    )
    top_k   = st.slider("Tampilkan Top-K", min_value=1, max_value=20, value=5)
    run_btn = st.button("🔍 Cari", use_container_width=True, type="primary")

    st.divider()
    st.subheader("Sketsa")
    if st.session_state.sketch_img:
        st.image(st.session_state.sketch_img, use_container_width=True,
                 caption=st.session_state.sketch_label)
        if st.session_state.gt_id:
            st.caption(f"Ground truth: `{st.session_state.gt_id}`")
    else:
        st.info("Belum ada sketsa")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    "**Cara pakai:** Klik **🎲 Coba Contoh** untuk memuat sketsa + deskripsi teks secara otomatis, "
    "atau upload sketsa sendiri lalu tulis deskripsi teks, kemudian klik **🔍 Cari**. "
    "Model menggabungkan informasi dari **sketsa** dan **teks** sekaligus untuk menemukan gambar Thangka yang paling relevan."
)
st.divider()

if run_btn:
    active_text = text_query.strip()
    if st.session_state.sketch_img is None:
        st.warning("Muat sketsa terlebih dahulu — klik Coba Contoh atau upload file.")
    elif not active_text:
        st.warning("Isi deskripsi teks terlebih dahulu.")
    else:
        with st.spinner("Menghitung retrieval..."):
            model, preprocess, device = load_model()
            results = retrieve(
                st.session_state.sketch_img, active_text, top_k,
                model, preprocess, device, gallery_embs, gallery_ids, level=level_choice
            )

        gt = st.session_state.gt_id
        st.subheader(f"Top-{top_k} Hasil Retrieval")
        if gt:
            hit_rank = next((r+1 for r, (iid, _) in enumerate(results) if iid == gt), None)
            if hit_rank:
                st.success(f"✓ Ground truth **{gt}** ditemukan di Rank #{hit_rank}")
            else:
                st.info(f"Ground truth **{gt}** tidak masuk Top-{top_k}")

        # ── Sketsa + hasil berdampingan ──────────────────────────────────────
        col_sketch, col_arrow, col_results = st.columns([1, 0.15, top_k])

        with col_sketch:
            st.markdown("**Input Kueri**")
            st.image(st.session_state.sketch_img, use_container_width=True, caption="Sketsa")
            st.caption(f"_{active_text[:120]}{'...' if len(active_text) > 120 else ''}_")

        with col_arrow:
            st.markdown("<div style='font-size:2rem; text-align:center; margin-top:80px'>→</div>",
                        unsafe_allow_html=True)

        with col_results:
            st.markdown("**Hasil Retrieval**")
            result_cols = st.columns(min(top_k, 5))
            for rank, (img_id, score) in enumerate(results):
                col = result_cols[rank % 5]
                img_path = id_to_path.get(img_id, '')
                is_gt = (img_id == gt)
                with col:
                    if os.path.exists(img_path):
                        st.image(Image.open(img_path).convert("RGB"), use_container_width=True)
                    else:
                        st.error(img_id)
                    label = f"**#{rank+1}** {'✓ GT' if is_gt else ''} · {score:.4f}"
                    st.caption(label)

        # ── Bar chart similarity ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Skor Kemiripan (Similarity)**")
        chart_data = pd.DataFrame({
            "Rank": [f"#{r+1} {'✓' if results[r][0]==gt else ''} {results[r][0]}" for r in range(len(results))],
            "Similarity": [s for _, s in results],
        })
        chart_data = chart_data.sort_values("Similarity")
        st.bar_chart(chart_data.set_index("Rank"), horizontal=True, color="#4F8BF9")

else:
    st.subheader("Galeri Thangka (419 gambar)")
    n_cols = 6
    rows = [gallery_ids[i:i+n_cols] for i in range(0, min(len(gallery_ids), 30), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, img_id in zip(cols, row):
            img_path = id_to_path.get(img_id, '')
            if os.path.exists(img_path):
                col.image(Image.open(img_path).convert("RGB"), use_container_width=True, caption=img_id)
    if len(gallery_ids) > 30:
        st.caption(f"...dan {len(gallery_ids)-30} gambar lainnya")
