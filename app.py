import streamlit as st
import os
from inference import load_model, generate_response

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindForge — Fine-tuned LLM",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080C14;
    color: #E0E8F0;
}
.stApp { background-color: #080C14; }

.hero {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #00D4FF;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.hero-sub {
    color: #4A6080;
    font-size: 0.9rem;
    margin-top: 4px;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

.info-card {
    background: #0D1520;
    border: 1px solid #1A2A3A;
    border-left: 3px solid #00D4FF;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
    color: #607080;
}

.chat-user {
    background: #0D2040;
    border: 1px solid #1A3A6A;
    border-radius: 12px 12px 4px 12px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    color: #A0C8E8;
    font-size: 0.9rem;
}

.chat-bot {
    background: #0D1520;
    border: 1px solid #1A2A3A;
    border-radius: 12px 12px 12px 4px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    color: #C0D8E8;
    font-size: 0.9rem;
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
}

.tag {
    display: inline-block;
    background: #001A2A;
    border: 1px solid #00D4FF33;
    color: #00D4FF;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    margin-right: 4px;
}

.stButton > button {
    background: #00D4FF !important;
    color: #080C14 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
}

.stButton > button:hover { opacity: 0.8 !important; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0D1520 !important;
    border: 1px solid #1A2A3A !important;
    color: #E0E8F0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}

hr { border-color: #1A2A3A !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero"> MindForge</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">FINE-TUNED GPT-2 · HUGGINGFACE TRANSFORMERS · CUSTOM Q&A</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <span class="tag">MODEL</span> GPT-2 Fine-tuned &nbsp;
    <span class="tag">TASK</span> Instruction Following &nbsp;
    <span class="tag">DOMAIN</span> AI & ML Q&A &nbsp;
    <span class="tag">FRAMEWORK</span> HuggingFace
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Load model ─────────────────────────────────────────────────────────────────
if st.session_state.model is None:
    model_path = "./fine_tuned_model"
    if os.path.exists(model_path):
        with st.spinner("Loading fine-tuned model..."):
            try:
                model, tokenizer = load_model(model_path)
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("✅ Fine-tuned model loaded!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
    else:
        st.warning("⚠️ Fine-tuned model not found. Please run `python train.py` first to train the model.")
        st.code("python train.py", language="bash")
        st.stop()

# ── Chat interface ────────────────────────────────────────────────────────────
st.markdown("---")

col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input(
        "Ask anything about AI & ML",
        placeholder="What is fine-tuning? What is a transformer?...",
        label_visibility="collapsed"
    )
with col2:
    ask_btn = st.button("Send →")

# ── Settings ──────────────────────────────────────────────────────────────────
with st.expander("⚙️ Generation Settings"):
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1,
                            help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max new tokens", 50, 300, 150, 10)

# ── Generate ──────────────────────────────────────────────────────────────────
if ask_btn and question and st.session_state.model:
    with st.spinner("Generating response..."):
        try:
            answer = generate_response(
                st.session_state.model,
                st.session_state.tokenizer,
                question,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="chat-user">👤 {chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot">⚡ {chat["answer"]}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#1A2A3A; font-size:0.75rem; font-family:'Space Mono',monospace;">
    GPT-2 · HuggingFace Transformers · Fine-tuned on Custom Q&A Data
</div>
""", unsafe_allow_html=True)
