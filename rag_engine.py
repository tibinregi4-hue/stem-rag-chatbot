import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama


# Paths
VECTORSTORE_PATH = "vectorstore"

# Embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def index_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )
    
    return len(chunks)

def ask_question(question):
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    
    llm = Ollama(model="mistral")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    docs = retriever.invoke(question)
    
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
        sources.append(f"{os.path.basename(source)} (page {page + 1})")
    
    sources = list(set(sources))
    
    return answer, sources