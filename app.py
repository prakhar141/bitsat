import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

st.title("🎯 BITSAT Question Solver")
st.markdown("**Powered by DeepSeek-R1-Distill-Qwen-1.5B-Q8_0**")

@st.cache_resource
def load_model():
    print("Downloading model from bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF ...")
    model_path = hf_hub_download(
        repo_id="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
    )
    print(f"Downloaded: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1e9:.2f} GB")
    print("Loading model into memory...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=True
    )
    print("Model ready!")
    return llm

with st.spinner("Loading model... (only happens once)"):
    llm = load_model()

# ── Functions ────────────────────────────────────────────────────────────────

def solve(question, opt_a, opt_b, opt_c, opt_d):
    if not question.strip():
        return "⚠️ Please enter a question."
    if not all([opt_a.strip(), opt_b.strip(), opt_c.strip(), opt_d.strip()]):
        return "⚠️ Please fill in all 4 options."

    prompt = f"""You are a BITSAT expert. Solve this multiple choice question.

Question: {question}

Options:
A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}

Think step by step and give the correct answer in this format:
Answer: X) <option text>"""

    response = llm(prompt, max_tokens=512, stop=["\n\n\n"], echo=False)
    output = response["choices"][0]["text"].strip()

    for line in output.split("\n"):
        if "answer:" in line.lower():
            return f"✅ {line.strip()}"

    lines = [l for l in output.split("\n") if l.strip()]
    return f"✅ {lines[-1]}" if lines else "⚠️ Could not determine answer."


def regenerate_options(question, opt_a, opt_b, opt_c, opt_d):
    if not question.strip():
        return opt_a, opt_b, opt_c, opt_d

    prompt = f"""Generate 4 multiple choice options for this BITSAT question. Exactly one must be correct. Format:
A) <option>
B) <option>
C) <option>
D) <option>

Question: {question}"""

    response = llm(prompt, max_tokens=200, stop=["\n\n"], echo=False)
    output = response["choices"][0]["text"].strip()
    options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("A)"):
            options["A"] = line[2:].strip()
        elif line.startswith("B)"):
            options["B"] = line[2:].strip()
        elif line.startswith("C)"):
            options["C"] = line[2:].strip()
        elif line.startswith("D)"):
            options["D"] = line[2:].strip()
    return options["A"], options["B"], options["C"], options["D"]


def regenerate_question(question, opt_a, opt_b, opt_c, opt_d):
    if not any([opt_a.strip(), opt_b.strip(), opt_c.strip(), opt_d.strip()]):
        return question

    prompt = f"""Write a clear BITSAT-style question for these 4 options. Return only the question.

A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}

Question:"""

    response = llm(prompt, max_tokens=150, stop=["\n\n", "A)"], echo=False)
    output = response["choices"][0]["text"].strip()
    return output if output else question


# ── UI ───────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("💡 Edit question → click **Sync Options** | Edit options → click **Sync Question** | Click **Solve** for answer")

question = st.text_area("📝 Question", value=st.session_state.get("question", ""), placeholder="Type your BITSAT question here...")
col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("A)", value=st.session_state.get("opt_a", ""), placeholder="Option A")
    opt_c = st.text_input("C)", value=st.session_state.get("opt_c", ""), placeholder="Option C")
with col2:
    opt_b = st.text_input("B)", value=st.session_state.get("opt_b", ""), placeholder="Option B")
    opt_d = st.text_input("D)", value=st.session_state.get("opt_d", ""), placeholder="Option D")

col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🔍 Solve", type="primary", use_container_width=True):
        with st.spinner("Thinking..."):
            result = solve(question, opt_a, opt_b, opt_c, opt_d)
        st.success(result)

with col_b:
    if st.button("🔄 Sync Options", use_container_width=True):
        with st.spinner("Regenerating options..."):
            a, b, c, d = regenerate_options(question, opt_a, opt_b, opt_c, opt_d)
        st.session_state["opt_a"] = a
        st.session_state["opt_b"] = b
        st.session_state["opt_c"] = c
        st.session_state["opt_d"] = d
        st.rerun()

with col_c:
    if st.button("🔄 Sync Question", use_container_width=True):
        with st.spinner("Regenerating question..."):
            new_q = regenerate_question(question, opt_a, opt_b, opt_c, opt_d)
        st.session_state["question"] = new_q
        st.rerun()
