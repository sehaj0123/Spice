import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

st.set_page_config(page_title="SPICE RAG Chatbot", page_icon="💬", layout="wide")

st.title("SPICE RAG Chatbot")
st.write("Ask questions about the SPICE solar project data.")

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Change file name if your dataset name is different
    df = pd.read_csv("merged_dataset.csv")
    return df

# -----------------------------------------------------------------------------
# 2. Convert dataframe rows into text chunks
# -----------------------------------------------------------------------------
@st.cache_data
def make_documents(df):
    documents = []

    for _, row in df.iterrows():
        text = (
            f"Date: {row.get('date', 'N/A')}. "
            f"Production: {row.get('Production', 'N/A')}. "
            f"Bissell total filled: {row.get('Bissell_total_filled', 'N/A')}. "
            f"Solar radiation: {row.get('solar_radiation', 'N/A')}. "
            f"Clear sky solar: {row.get('solar_clear_sky', 'N/A')}. "
            f"NASA temperature: {row.get('temperature_nasa', 'N/A')}. "
            f"Mean temperature: {row.get('Mean Temp (°C)', 'N/A')}. "
            f"Rain: {row.get('Total Rain (mm)', 'N/A')}. "
            f"Snow: {row.get('Total Snow (cm)', 'N/A')}. "
            f"Pool price: {row.get('pool_price', 'N/A')}."
        )
        documents.append(text)

    return documents

# -----------------------------------------------------------------------------
# 3. Load embedding model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------------------------------------
# 4. Create embeddings
# -----------------------------------------------------------------------------
@st.cache_resource
def create_embeddings(documents):
    embedder = load_embedder()
    embeddings = embedder.encode(documents, convert_to_tensor=True)
    return embeddings

# -----------------------------------------------------------------------------
# 5. Load generation model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_generator():
    # Smaller model is better for Streamlit deployment
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

# -----------------------------------------------------------------------------
# 6. Retrieve top relevant chunks
# -----------------------------------------------------------------------------
def retrieve_context(query, documents, embeddings, top_k=3):
    embedder = load_embedder()
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=min(top_k, len(documents)))

    retrieved_docs = [documents[idx] for idx in top_results.indices]
    return "\n\n".join(retrieved_docs), top_results

# -----------------------------------------------------------------------------
# 7. Generate answer
# -----------------------------------------------------------------------------
def generate_answer(query, context):
    generator = load_generator()

    prompt = f"""
You are a helpful chatbot for the SPICE solar energy project.

Use only the provided context to answer the question.
If the answer is not available in the context, say:
"I could not find that information in the uploaded data."

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return result[0]["generated_text"].strip()

# -----------------------------------------------------------------------------
# 8. Main app
# -----------------------------------------------------------------------------
try:
    df = load_data()
    documents = make_documents(df)
    embeddings = create_embeddings(documents)

    st.success("SPICE data loaded successfully.")

    with st.expander("See dataset preview"):
        st.dataframe(df.head())

    user_query = st.text_input("Ask a question about your solar data:")

    example_questions = [
        "What does the dataset say about solar production?",
        "How does solar radiation relate to production?",
        "What information is available about pool price?",
        "Which weather factors appear in the dataset?",
        "Summarize the SPICE solar data."
    ]

    st.write("**Example questions:**")
    for q in example_questions:
        st.write(f"- {q}")

    if user_query:
        with st.spinner("Retrieving answer..."):
            context, top_results = retrieve_context(user_query, documents, embeddings, top_k=3)
            answer = generate_answer(user_query, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            st.write(context)

except FileNotFoundError:
    st.error("The file 'merged_dataset.csv' was not found. Please check the file name and location.")
except Exception as e:
    st.error(f"An error occurred: {e}")
