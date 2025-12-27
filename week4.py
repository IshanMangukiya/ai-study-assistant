import streamlit as st
import os
import numpy as np
import faiss
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("📚 AI Study Assistant – Week 4 (Vector Search)")

# ---------- Helper Functions ----------

def extract_pdf_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# ---------- UI ----------

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_pdf_text(uploaded_file)
        chunks = chunk_text(text)

        embeddings = [get_embedding(chunk) for chunk in chunks]

        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        st.success("✅ PDF indexed successfully")

    question = st.text_input("Ask a question from the PDF")

    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching best answer... 🤔"):
                q_embedding = get_embedding(question)
                D, I = index.search(np.array([q_embedding]), k=3)

                context = "\n".join([chunks[i] for i in I[0]])

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Answer only using the given context."},
                        {"role": "user", "content": f"Context:\n{context}"},
                        {"role": "user", "content": f"Question: {question}"}
                    ]
                )

                answer = response.choices[0].message.content
                st.subheader("📌 Answer")
                st.write(answer)
