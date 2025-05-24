import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 1) Device setup ────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── 2) Load model & tokenizer with caching ─────────
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert_model",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained("bert_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ── 3) Label mapping ───────────────────────────────
label_map = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4"
}

# ── 4) Streamlit UI ────────────────────────────────
st.title("TCFD Recommendations")
st.header("BERT-based Text Classification App")
st.subheader("CS 180 THR - Domingo, Ramos, Senatin")

text_input = st.text_area("Enter your text:")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        encoded = tokenizer(
            text_input,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        model = model.to(device)  # Safely move model only once here

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.success(f"Predicted class: {label_map[pred]}")
        st.write("Class probabilities:")
        for i, prob in enumerate(probs[0]):
            st.write(f"{label_map[i]}: {prob.item():.2%}")