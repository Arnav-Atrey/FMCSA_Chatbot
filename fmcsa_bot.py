import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set the GROQ API key securely
os.environ["GROQ_API_KEY"] = "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"

# Initialize the language model
llm = ChatGroq(model="llama3-8b-8192", max_tokens=4096)

# Initialize the HuggingFace embeddings with the model name
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define a simple Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Streamlit app
st.title("FMCSA Q&A with LLM")

# Initialize Chroma vector store
vectorstore_path = r"C:\Users\Arnav\Desktop\PES\internship\AI_BOT_PDFS\chroma_db"
vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)

# Define tabs
tab1, tab2 = st.tabs(["Upload PDF", "Ask Questions"])

with tab1:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    if uploaded_file:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_pdf_path = tmp_file.name

        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(temp_pdf_path)

        # Load and chunk the text
        doc = Document(page_content=pdf_text) # Create a Document object
        docs = [doc]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Store chunks in Chroma DB
        vectorstore.add_documents(documents=splits)#time consuming
        st.success("PDF content has been stored in Chroma DB successfully.")

with tab2:
    st.header("Ask Questions on Stored Data")

    # Retrieve and generate using the relevant snippets of the PDF
    retriever = vectorstore.as_retriever()
    prompt_template = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Custom prompt to instruct the AI about its role
    custom_prompt = """
    You are an AI assistant. Your task is to help the driver by answering their questions based on the content of
    the provided PDF document. Retrieve the most relevant information from the PDF and provide a detailed answer
    to the user's query. Also provide citations from where data has been picked.

    Context: {context}

    Question: {question}

    Detailed Answer:
    """
    # Function to format the prompt with context and question
    def create_prompt(context, question):
        return custom_prompt.format(context=context, question=question)

    # Take user input and invoke the chain
    question = st.text_input("Ask a question about the stored PDF content:")
    if st.button("Get Answer") and question:
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question) #time consuming
        #st.write("Retrieved Relevant Docs")
        formatted_context = format_docs(relevant_docs)
        #st.write("created formatted context")
        final_prompt = create_prompt(context=formatted_context, question=question)
        #st.write("created final prompt")
        
        # Generate the answer using the final prompt
        result = llm.invoke(final_prompt)
        st.write("Answer")
        st.write(result.content)  # Access and display only the content attribute