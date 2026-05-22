import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tempfile
import hashlib
from dotenv import load_dotenv

# =========================
# LANGCHAIN IMPORTS
# =========================
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="PDF Comparison AI",
    page_icon="📄",
    layout="wide"
)

# =========================
# ENV VARIABLES
# =========================
load_dotenv()

# =========================
# STREAMLIT SECRETS
# =========================
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_CHAT_DEPLOYMENT = st.secrets["AZURE_OPENAI_CHAT_DEPLOYMENT"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = ""


# =========================
# AZURE OPENAI
# =========================
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
    temperature=0
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
)

# =========================
# FUNCTIONS
# =========================
def generate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_pdf(file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

        tmp_file.write(file.getbuffer())

        temp_path = tmp_file.name

    loader = PyMuPDFLoader(temp_path)

    docs = loader.load()

    return docs, temp_path


def chunk_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)


def create_vectorstore(chunks):

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


def compare_pdfs(pdf1_docs, pdf2_docs):

    pdf1_text = "\n".join([doc.page_content for doc in pdf1_docs])
    pdf2_text = "\n".join([doc.page_content for doc in pdf2_docs])

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI PDF Comparison Assistant.

Compare the following two PDF documents.

Provide:

1. Summary of PDF 1
2. Summary of PDF 2
3. Similarities
4. Differences
5. Important changes
6. Final conclusion

PDF 1:
{pdf1}

PDF 2:
{pdf2}
"""
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "pdf1": pdf1_text[:15000],
        "pdf2": pdf2_text[:15000]
    })

    return response


def ask_question(question, vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question ONLY from the provided context.

Question:
{question}

Context:
{context}
"""
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "question": question,
        "context": context
    })

    return response


# =========================
# UI
# =========================
st.title("📄 PDF Comparison AI")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    pdf1 = st.file_uploader(
        "Upload First PDF",
        type=["pdf"],
        key="pdf1"
    )

with col2:

    pdf2 = st.file_uploader(
        "Upload Second PDF",
        type=["pdf"],
        key="pdf2"
    )

# =========================
# COMPARE BUTTON
# =========================
if st.button("Compare PDFs"):

    if pdf1 and pdf2:

        with st.spinner("Loading PDFs..."):

            pdf1_docs, path1 = load_pdf(pdf1)
            pdf2_docs, path2 = load_pdf(pdf2)

        with st.spinner("Comparing documents..."):

            result = compare_pdfs(
                pdf1_docs,
                pdf2_docs
            )

            st.session_state.comparison_result = result

            chunks = chunk_documents(
                pdf1_docs + pdf2_docs
            )

            vectorstore = create_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore

        st.success("Comparison completed!")

    else:
        st.warning("Please upload both PDFs")

# =========================
# RESULTS
# =========================
if st.session_state.comparison_result:

    st.subheader("📊 Comparison Result")

    st.write(st.session_state.comparison_result)

    st.markdown("---")

    st.subheader("💬 Ask Questions About PDFs")

    question = st.chat_input(
        "Ask question about uploaded PDFs"
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):

            answer = ask_question(
                question,
                st.session_state.vectorstore
            )

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

# =========================
# SIDEBAR HISTORY
# =========================
with st.sidebar:

    st.title("📜 Chat History")

    if len(st.session_state.chat_history) == 0:

        st.info("No chat history available")

    else:

        for item in st.session_state.chat_history:

            with st.expander(item["question"][:40]):

                st.write("Question:")
                st.write(item["question"])

                st.write("Answer:")
                st.write(item["answer"])
