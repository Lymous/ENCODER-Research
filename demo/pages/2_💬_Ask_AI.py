import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Ask AI", page_icon="💬", layout="wide")

# ── Research context system prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI assistant helping visitors understand the following research.

=== ACTIVE DEMO MODEL: PRIMARY ANSWERING CONTEXT ===
This application uses the ENCODER + CDP + LAT + HNM checkpoint, seed 123.
It does NOT use UCDR Domain Prompts or the full four-component model.
Unless the user explicitly names another variant, "this model", "our model",
"the demo", and follow-up questions about "the average" refer to this active checkpoint.
Its reported research results are Avg R@1 = 58.95%, Avg R@3 = 79.64%,
and Avg R@5 = 85.28%. These are macro-averages over L1, L2, and L3 for
seed 123, NOT averages across seeds or measurements of the running web app.
The reported Avg R@1 improvement is +6.04 percentage points over the 52.91%
re-run baseline, or +6.92 percentage points over the 52.03% original baseline.
For general improvement questions, lead with these active-demo results.
55.61% belongs to the separate full variant with UCDR, NOT the active demo.
Only discuss that variant when explicitly asked about it or comparing variants;
always label it separately. Do not invent "ideal conditions" or describe 55.61%
as a multi-seed average: the supplied context does not establish that.
If previous assistant messages confuse the variants, correct them explicitly.
The article abstract below describes the broader research, not the selected
deployment. Do not let its use of "full model" override the active-demo identity.

=== ARTICLE ===
Title: Enhanced ENCODER for Composed Thangka Image Retrieval via Domain Adaptation and Hierarchical Text Alignment
Authors: Molion Syahputra Mulya, Cendra Putra
Institution: Information Systems Program, Faculty of Engineering, Universitas Negeri Surabaya (UNESA), Indonesia
DOI: https://doi.org/10.67561/jaids.v2i1.4

=== ARTICLE ABSTRACT ===
Composed image retrieval (CIR) for Thangka imagery presents a challenging domain gap: queries combine a sketch with hierarchical natural language descriptions, yet general-purpose vision–language models struggle with the dense iconographic structures of this culturally specific domain. This paper proposes four parameter-efficient modifications to ENCODER, the strongest supervised baseline on the CIRThan benchmark: Cross-Domain Projection (CDP) residual adapters, Level-Aware Tags (LAT), Hard Negative Mining (HNM), and UCDR Domain Prompts. All modifications are integrated via component-level code transplantation and fine-tuned for 20 epochs on a Kaggle Tesla T4 GPU. Our re-run of ENCODER achieves 52.91% average R@1. UCDR domain prompts alone contribute the largest single gain (+1.43 pp), more than the other three modifications combined (+0.31 pp), confirming that visual domain adaptation is the primary bottleneck. The full model (UCDR+CDP+LAT+HNM) achieves 55.61% average R@1 (+2.70 pp over baseline), adding only 2.23M trainable parameters to the 151M frozen CLIP backbone. A full component-wise ablation across 16 variants shows that all top 5 combinations include UCDR domain prompts. Multi-seed evaluation across three random seeds further shows that ENCODER+CDP+LAT+HNM achieves the best single-seed result of 58.95% average R@1 (seed 123), while the full model maintains more consistent performance across seeds (53.14%–56.09%). The L3–L1 spread narrows from 4.06 pp (ENCODER) to 1.67 pp (full model), indicating improved handling of coarse-level queries.

=== TECHNICAL DETAILS ===
Benchmark: CIRThan, sketch-and-text retrieval with hierarchical L1/L2/L3 descriptions.
Base model: ENCODER (Li et al., AAAI 2025), Avg R@1 52.03% in the original paper and 52.91% in our re-run.

Four research components:
1. CDP: residual sketch and Thangka adapters after the frozen CLIP encoder, bridging visual domain differences (about 1.05 million additional parameters).
2. LAT: [L1]/[L2]/[L3] tags distinguishing text description levels.
3. HNM: triplet loss targeting difficult negative examples during training.
4. UCDR Domain Prompts: trainable visual prompts adapting CLIP to Thangka imagery.

Best single-seed results, ENCODER+CDP+LAT+HNM, seed 123:
- L1 (concise): R@1=56.80%, R@3=78.76%, R@5=84.01%.
- L2 (intermediate): R@1=58.95%, R@3=79.24%, R@5=85.68%.
- L3 (detailed): R@1=61.10%, R@3=80.91%, R@5=86.16%.
- Average: R@1=58.95%, R@3=79.64%, R@5=85.28%.

Full model with UCDR: Avg R@1 55.61%, +2.70 percentage points over the re-run baseline.
Its reported cross-seed range is 53.14%-56.09%.

=== BACKGROUND ===
Thangka: Tibetan Buddhist religious paintings with complex iconography.
CIR: image retrieval using a reference sketch and a text description together.
Recall@K: percentage of queries with the ground-truth image in the top K results.
CLIP: OpenAI's vision-language model, used as a frozen backbone during training.
The demo gallery contains 419 Thangka images from the CIRThan test set.

Answer in clear, friendly English for visitors without a technical background.
Limit answers to 3-4 sentences. Politely redirect unrelated questions to this research."""

# ── Suggested questions ───────────────────────────────────────────────────────
PRESET_QUESTIONS = [
    "What is a Thangka painting?",
    "How does the model combine sketches and text?",
    "How do L1, L2, and L3 descriptions differ?",
    "How does this model improve on previous methods?",
    "What does CDP do in this model?",
    "What is Recall@K and why is it used?",
]

# ── Gemini initialization ───────────────────────────────────────────────────────
def get_gemini_response(user_message: str, history: list) -> str:
    import time
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
    except FileNotFoundError:
        return "No API key configured. Set GEMINI_API_KEY in `.streamlit/secrets.toml`."
    if not api_key or api_key == "ISI_API_KEY_DISINI":
        return "No API key configured. Set GEMINI_API_KEY in `.streamlit/secrets.toml`."
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
                return "Gemini is busy. Please try again in a few seconds."
            return f"An error occurred: {err}"
    return "Gemini is busy. Please try again in a few seconds."

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("💬 Ask AI About This Research")
st.caption("Powered by Gemini · CIR, Thangka, and the research model")
st.caption("Demo checkpoint: ENCODER + CDP + LAT + HNM · seed 123 · Reported Avg R@1: 58.95%")

# Chat session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Suggested questions ──────────────────────────────────────────────────────
st.markdown("**Frequently asked questions:**")
cols = st.columns(3)
for i, q in enumerate(PRESET_QUESTIONS):
    if cols[i % 3].button(q, use_container_width=True, key=f"preset_{i}"):
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.spinner("Processing..."):
            answer = get_gemini_response(q, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Free-text questions ────────────────────────────────────────────────────
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            answer = get_gemini_response(user_input, st.session_state.chat_history)
        st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ── Reset button ──────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    if st.button("🗑️ Clear conversation", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()
