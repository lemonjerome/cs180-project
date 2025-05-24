import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "bert_model"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model config
config = AutoConfig.from_pretrained(MODEL_PATH)

# Initialize model in empty state on the target device
model = AutoModelForSequenceClassification.from_config(config).to_empty(device=device)

# Load the weights manually from the saved state dict
model.load_state_dict(torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=device))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Set model to eval mode
model.eval()

# Label mapping (modify this based on your actual class names)
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