# Import Libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import faiss
import numpy as np
import streamlit as st

# Load the BillSum dataset
dataset = load_dataset("billsum", split="ca_test")

# Initialize models
sbert_model = SentenceTransformer("all-mpnet-base-v2")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Prepare data and build FAISS index
texts = dataset["text"][:100]  # Limiting to 100 samples for speed
case_embeddings = sbert_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
index = faiss.IndexFlatL2(case_embeddings.shape[1])
index.add(np.array(case_embeddings.cpu()))

# Define retrieval and summarization functions
def retrieve_cases(query, top_k=3):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    _, indices = index.search(np.array([query_embedding.cpu()]), top_k)
    return [(texts[i], i) for i in indices[0]]

def summarize_text(text):
    inputs = t5_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
def main():
    st.title("Legal Case Summarizer")
    query = st.text_input("Enter your case search query here:")
    top_k = st.slider("Number of similar cases to retrieve:", 1, 5, 3)

    if st.button("Search"):
        results = retrieve_cases(query, top_k=top_k)
        for i, (case_text, index) in enumerate(results):
            st.subheader(f"Case {i+1}")
            st.write("**Original Text:**", case_text)
            summary = summarize_text(case_text)
            st.write("**Summary:**", summary)

if __name__ == "__main__":
    main()

# Run Streamlit app within Colab
