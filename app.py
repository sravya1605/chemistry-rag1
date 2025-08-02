import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OpenAI API key not found. Please check your .env file.")
    st.stop()

st.set_page_config(page_title="Chemistry RAG", layout="wide")
st.title("🔬 Chemistry RAG: Ask a Question from Scientific PDF")
query = st.text_input("🔍 Enter your chemistry question:")

@st.cache_resource
def load_documents():
    docs = []
    data_path = "data"
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            docs.extend(loader.load())
    return docs

if query:
    with st.spinner("🧪 Thinking... Reading PDFs... Generating response..."):
        raw_docs = load_documents()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(raw_docs)

        embeddings = HuggingFaceEmbeddings()
        vectordb = Chroma.from_documents(chunks, embedding=embeddings)

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever
        )

        result = qa_chain.run(query)
        st.subheader("🧠 Answer:")
        st.write(result)
