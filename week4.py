# ai_study_assistant.py
import streamlit as st
from annoy import AnnoyIndex
import numpy as np
from PyPDF2 import PdfReader
import openai
import os
from dotenv import load_dotenv

# -----------------------------
# Load OpenAI API key
# -----------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("AI Study Assistant 🤖")

# -----------------------------
# PDF Upload (Optional)
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF (optional)", type="pdf")

chunks = []
vectors = []
index = None
dim = 128  # default dimension if no PDF

if uploaded_file:
    # Extract text from PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    st.success("PDF uploaded successfully!")

    # Split text into chunks
    chunks = [line for line in text.split("\n") if line.strip()]

    # Generate embeddings for chunks
    vectors = []
    for chunk in chunks:
        response = openai.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        vectors.append(response['data'][0]['embedding'])

    dim = len(vectors[0])

    # Build Annoy Index
    index = AnnoyIndex(dim, 'angular')
    for i, vec in enumerate(vectors):
        index.add_item(i, vec)
    index.build(10)  # number of trees

# -----------------------------
# Search Bar (Always Visible)
# -----------------------------
user_question = st.text_input("Ask a question:")

if st.button("Ask"):
    if not user_question:
        st.warning("Please enter a question")
    else:
        if uploaded_file and index is not None:
            # Use PDF content to answer
            response = openai.embeddings.create(
                input=user_question,
                model="text-embedding-3-small"
            )
            user_vector = response['data'][0]['embedding']
            nearest_ids = index.get_nns_by_vector(user_vector, 1)
            answer = chunks[nearest_ids[0]]
            st.success("Answer from PDF: " + answer)
        else:
            # No PDF: Use GPT to answer general question
            gpt_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI study assistant."},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.7,
            )
            answer = gpt_response.choices[0].message.content
            st.success("AI Response: " + answer)



