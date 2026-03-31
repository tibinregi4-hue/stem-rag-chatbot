import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama

# Paths
VECTORSTORE_PATH = "vectorstore"
UPLOADS_PATH = "uploads"

# Embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def index_pdf(pdf_path, subject="General"):
    """Read PDF, split into chunks, ADD to existing ChromaDB"""
    
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    # Add subject tag to each chunk metadata
    for doc in documents:
        doc.metadata["subject"] = subject
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    # ADD to existing ChromaDB instead of overwriting
    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTORSTORE_PATH
        )
    
    return len(chunks)

def get_uploaded_files():
    """Return list of uploaded PDF filenames"""
    if not os.path.exists(UPLOADS_PATH):
        return []
    files = [f for f in os.listdir(UPLOADS_PATH) if f.endswith(".pdf")]
    return files

def ask_question(question, subject_filter=None):
    """Find relevant chunks and ask Mistral"""
    
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    
    llm = Ollama(model="mistral")
    
    # Apply subject filter if selected
    if subject_filter and subject_filter != "All":
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"subject": subject_filter}
            }
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    docs = retriever.invoke(question)
    
    if not docs:
        return "I could not find any relevant information in the uploaded documents.", []
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a helpful STEM teaching assistant.
Answer the question based ONLY on the following context from uploaded documents.
If the answer is not in the context, say "I could not find this in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""
    
    answer = llm.invoke(prompt)
    
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        subject = doc.metadata.get("subject", "General")
        sources.append(f"{os.path.basename(source)} (page {page + 1}) [{subject}]")
    
    sources = list(set(sources))
    
    return answer, sources