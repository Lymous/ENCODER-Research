import streamlit as st
import pandas as pd

st.set_page_config(page_title="About - CIR Thangka", page_icon="📖", layout="wide")
st.title("📖 About This Research")
st.caption("Composed Image Retrieval for Thangka Images Using ENCODER")
st.divider()

st.header("🔍 What Is Composed Image Retrieval (CIR)?")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
Composed Image Retrieval (CIR) combines **two inputs**: a **reference image**
(sketch) and a **text description**. The model uses both to find the most
relevant target image in a gallery.

Example: *"Find a Thangka painting similar to this sketch, with a central
Buddha figure and a lotus background."*

Combining images and text allows more specific and expressive queries than
using either input alone.
""")
with col2:
    st.info("**Sketch** (reference image)\n\n+ **Text** (modification description)\n\n↓\n\n**Relevant Thangka image**")
st.divider()

st.header("🖼️ CIRThan Dataset & Thangka Images")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Thangka** paintings are Tibetan Buddhist religious artworks with complex
visual and symbolic structures. Figures, colors, and attributes carry
distinct iconographic meanings.

**CIRThan** provides paired Thangka images, line sketches, and three levels
of text descriptions (L1, L2, L3). The training split contains **1,868**
pairs, and the test split contains **419** pairs.
""")
with col2:
    st.markdown("""
**Three Text Description Levels:**

| Level | Description | Illustrative length |
|---|---|---|
| **L1** | Concise | ~10 words |
| **L2** | Intermediate | ~30 words |
| **L3** | Detailed | ~50 words |

Higher levels describe the figures' iconographic attributes more specifically.
""")
st.divider()

st.header("⚙️ Research Model")
st.markdown("""
This research extends **ENCODER** (Li et al., 2025) with four components
to address the challenges of the Thangka domain:
""")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**1. CDP (Cross-Domain Projection)**
Bridges visual differences between sketches and Thangka images.
Two residual adapters follow the frozen CLIP encoder, one for each modality.
*(Approximately 1.05 million additional parameters.)*

**2. LAT (Level-Aware Tags)**
Adds L1/L2/L3 markers to distinguish description levels in the shared text
representation space.
""")
with col2:
    st.markdown("""
**3. HNM (Hard Negative Mining)**
Uses triplet loss to target difficult negative examples in the embedding
space, helping the model learn from similar but incorrect matches and
improving stability across random seeds.

**4. UCDR Domain Prompts**
Uses trainable prompt tokens, following Visual Prompt Tuning, to adapt
CLIP's visual encoder to the characteristics of Thangka imagery.
""")
st.divider()

st.header("📊 Evaluation Results on CIRThan")
st.markdown("Metric: **Recall@K (R@K, %)**, the percentage of queries whose ground-truth image appears in the top K results.")
col_tbl, col_gap, col_highlight = st.columns([3, 0.3, 1.5])
with col_tbl:
    df = pd.DataFrame({
        "Model": [
            "COMBINER (Baldrati et al.)", "CLIP4CIR (Baldrati et al.)",
            "BiLLIP4CIR (Liu et al.)", "CALA (Jiang et al.)",
            "ENCODER baseline (Li et al.)", "ENCODER re-run (ours)",
            "✨ ENCODER+CDP+LAT+HNM (ours)",
        ],
        "Avg R@1": ["≤30%", "≤30%", "≤30%", "≤30%", "52.03%", "52.91%", "58.95%"],
        "Avg R@3": ["n/a", "n/a", "n/a", "n/a", "75.50%", "74.30%", "79.64%"],
        "Avg R@5": ["n/a", "n/a", "n/a", "n/a", "84.17%", "81.07%", "85.28%"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Results on the CIRThan test split. Avg is the macro-average across L1, L2, and L3.")
with col_highlight:
    st.metric("Avg R@1 (Best)", "58.95%", delta="+6.92 pp vs baseline")
    st.metric("Avg R@3 (Best)", "79.64%", delta="+4.14 pp vs baseline")
    st.metric("Avg R@5 (Best)", "85.28%", delta="+1.11 pp vs baseline")
st.divider()

st.header("📈 Per-Level Results (Best Model)")
st.markdown("Best model: **ENCODER+CDP+LAT+HNM**, seed 123.")
df_level = pd.DataFrame({
    "Level": ["L1 (Concise)", "L2 (Intermediate)", "L3 (Detailed)", "Average"],
    "R@1 (%)": ["56.80", "58.95", "61.10", "58.95"],
    "R@3 (%)": ["78.76", "79.24", "80.91", "79.64"],
    "R@5 (%)": ["84.01", "85.68", "86.16", "85.28"],
})
st.dataframe(df_level, use_container_width=True, hide_index=True)
st.caption("More detailed L3 descriptions achieve higher R@K, supporting the use of hierarchical text information.")
st.divider()
st.markdown("_This research forms part of the undergraduate thesis in Information Systems at Universitas Negeri Surabaya (UNESA). Author: Molion Syahputra Mulya._")
