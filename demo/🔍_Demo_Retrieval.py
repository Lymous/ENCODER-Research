"""
CIR Thangka Demo — Streamlit app
Run:
  streamlit run demo/🔍_Demo_Retrieval.py
"""

import os, sys, json, random
import numpy as np
import torch
import streamlit as st
import pandas as pd
from PIL import Image
from inference import DATASET_DIR as DATA_PATH, load_model as load_inference_model, retrieve
from media import ensure_media

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR    = os.path.dirname(os.path.abspath(__file__))
ENCODER_DIR = os.path.join(ROOT, 'ENCODER_repo_new')
DATASET_DIR = str(DATA_PATH)
CHECKPOINT  = os.path.join(ROOT, 'results', 'enhanced_best 123.pt')
CACHE_DIR   = os.path.join(DEMO_DIR, 'gallery_cache')


def media_path(path):
    try:
        return ensure_media(path, DATASET_DIR)
    except Exception:
        st.warning("The dataset host is temporarily unavailable. Please try again later.")
        return path

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
    return load_inference_model('cpu' if os.environ.get('ENCODER_CLOUD') == '1' else None)


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
st.caption("Model: ENCODER + CDP + LAT + HNM (seed 123) · Gallery: 419 Thangka images")

try:
    gallery_ids, gallery_embs, id_to_path = load_gallery_meta()
    test_samples = load_test_samples()
except FileNotFoundError:
    st.error("Demo assets are missing. Run python scripts/setup_demo.py and follow the dataset steps in README.")
    st.stop()

with st.sidebar:
    st.header("Query Input")

    # ── Test example button ──────────────────────────────────────────────────
    if st.button("🎲 Try a Test Example", use_container_width=True):
        sample = random.choice(test_samples)
        sketch_num  = sample['sketch_id'].replace('sketch', '')
        sketch_path = os.path.join(DATASET_DIR, 'sketch', f"{sketch_num}.png")
        sketch_path = media_path(sketch_path)
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
            st.warning(f"Sketch file not found: {sketch_path}")

    st.divider()

    # ── Manual upload ───────────────────────────────────────────────────────
    sketch_file = st.file_uploader("Or upload your own sketch", type=["png", "jpg", "jpeg"])
    if sketch_file:
        st.session_state.sketch_img   = Image.open(sketch_file).convert('RGB')
        st.session_state.gt_id        = None
        st.session_state.sketch_label = sketch_file.name
        st.session_state.levels       = None

    # ── Level selector for test examples ───────────────
    level_choice = 1
    if st.session_state.levels:
        st.markdown("**Text description level**")
        level_choice = st.radio(
            "Level",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "L1 — Concise",
                2: "L2 — Intermediate",
                3: "L3 — Detailed",
            }[x],
            horizontal=True,
            key="level_radio",
            label_visibility="collapsed",
        )
        selected_text = st.session_state.levels[level_choice]
        # Sync text_input when the description level changes
        if st.session_state.get("text_input") != selected_text:
            st.session_state["text_input"] = selected_text
            st.rerun()
    else:
        level_choice = st.selectbox("Description level", [1, 2, 3], format_func=lambda x: f"L{x}")

    text_query = st.text_area(
        "Text description",
        placeholder="Example: a thangka with a central Buddha figure surrounded by lotus flowers",
        key="text_input",
    )
    top_k   = st.slider("Number of results (Top-K)", min_value=1, max_value=20, value=5)
    run_btn = st.button("🔍 Search", use_container_width=True, type="primary")

    st.divider()
    st.subheader("Sketch")
    if st.session_state.sketch_img:
        st.image(st.session_state.sketch_img, use_container_width=True,
                 caption=st.session_state.sketch_label)
        if st.session_state.gt_id:
            st.caption(f"Ground truth: `{st.session_state.gt_id}`")
    else:
        st.info("No sketch selected")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    "**Query:** "
    ""
    "A **sketch** and **text description** represent the target Thangka image."
)
st.divider()

if run_btn:
    active_text = text_query.strip()
    if st.session_state.sketch_img is None:
        st.warning("Select a test example or upload a sketch first.")
    elif not active_text:
        st.warning("Enter a text description first.")
    else:
        with st.spinner("Retrieving images..."):
            model, preprocess, device = load_model()
            results = retrieve(
                st.session_state.sketch_img, active_text, top_k,
                model, preprocess, device, gallery_embs, gallery_ids, level=level_choice
            )

        gt = st.session_state.gt_id
        st.subheader(f"Top-{top_k} Retrieval Results")
        if gt:
            hit_rank = next((r+1 for r, (iid, _) in enumerate(results) if iid == gt), None)
            if hit_rank:
                st.success(f"✓ Ground truth **{gt}** found at rank #{hit_rank}")
            else:
                st.info(f"Ground truth **{gt}** is not in the Top-{top_k}")

        # ── Sketch and results side by side ──────────────────────────────────────
        col_sketch, col_arrow, col_results = st.columns([1, 0.15, top_k])

        with col_sketch:
            st.markdown("**Query Input**")
            st.image(st.session_state.sketch_img, use_container_width=True, caption="Sketch")
            st.caption(f"_{active_text[:120]}{'...' if len(active_text) > 120 else ''}_")

        with col_arrow:
            st.markdown("<div style='font-size:2rem; text-align:center; margin-top:80px'>→</div>",
                        unsafe_allow_html=True)

        with col_results:
            st.markdown("**Retrieval Results**")
            result_cols = st.columns(min(top_k, 5))
            for rank, (img_id, score) in enumerate(results):
                col = result_cols[rank % 5]
                img_path = id_to_path.get(img_id, '')
                if img_path:
                    img_path = media_path(img_path)
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
        st.markdown("**Similarity Scores**")
        chart_data = pd.DataFrame({
            "Rank": [f"#{r+1} {'✓' if results[r][0]==gt else ''} {results[r][0]}" for r in range(len(results))],
            "Similarity": [s for _, s in results],
        })
        chart_data = chart_data.sort_values("Similarity")
        st.bar_chart(chart_data.set_index("Rank"), horizontal=True, color="#4F8BF9")

else:
    st.subheader("Thangka Gallery (419 images)")
    n_cols = 6
    gallery_limit = 6 if os.environ.get('ENCODER_CLOUD') == '1' else 30
    rows = [gallery_ids[i:i+n_cols] for i in range(0, min(len(gallery_ids), gallery_limit), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, img_id in zip(cols, row):
            img_path = id_to_path.get(img_id, '')
            if img_path:
                img_path = media_path(img_path)
            if os.path.exists(img_path):
                col.image(Image.open(img_path).convert("RGB"), use_container_width=True, caption=img_id)
    if len(gallery_ids) > gallery_limit:
        st.caption(f"...and {len(gallery_ids)-gallery_limit} more images")
