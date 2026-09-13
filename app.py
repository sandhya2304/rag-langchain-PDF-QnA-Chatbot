
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore

load_dotenv()


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# Session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False


# PDF processing
def document_process(path):

    # Document loading
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Text splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    # Vector store
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # Save vector database in session
    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True


# Page UI
st.title("Document QnA Chatbot")
st.subheader("Ask Anything from Your PDF")


# PDF upload
if not st.session_state.document_uploaded:

    file = st.file_uploader(
        label="Select your PDF File",
        type="pdf"
    )

    if file:

        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing Document..."):

            document_process("./uploaded_document.pdf")

        st.success(
            "Document Processed Successfully! "
            "You can now ask questions about the document."
        )

        st.rerun()


# Chat UI
if st.session_state.document_uploaded:

    st.success("Document is ready. Ask your question!")

    question = st.text_input(
        "Ask a question about your document:",
        key="question_input"
    )

    if question:

        # Retrieve relevant chunks
        docs = st.session_state.vector_db.similarity_search(
            question,
            k=3
        )

        # Create context
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # Prompt
        prompt = f"""
        Answer the question using only the context provided below.

        Context:
        {context}

        Question:
        {question}

        If the answer is not available in the context,
        say "I don't know based on the provided document."
        """

        # Gemini response
        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)