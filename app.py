import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load model and tokenizer
MODEL_PATH = "bert_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Use MPS or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# Label mapping (modify this based on your classes)
label_map = {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4"}

# Streamlit UI
st.title("TCFD Recommendations")
st.header("BERT-based Text Classification App")
st.subheader("CS 180 THR - Domingo, Ramos, Senatin")
text_input = st.text_area("Enter your text:")

if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input
        encoded = tokenizer(
            text_input,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.success(f"Predicted class: {label_map[pred]}")
        st.write("Class probabilities:")
        for i, prob in enumerate(probs[0]):
            st.write(f"{label_map[i]}: {prob.item():.2%}")