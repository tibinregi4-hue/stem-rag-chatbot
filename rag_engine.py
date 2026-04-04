import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama

VECTORSTORE_PATH = "vectorstore"
UPLOADS_PATH = "uploads"


def get_embeddings():
    """Create fresh embeddings instance each time"""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def detect_language(text):
    """Strictly detect if question is German or English"""
    german_words = [
        "wie", "was", "warum", "erkläre", "erklär", "ich", "bitte",
        "können", "ist", "sind", "lehren", "unterricht", "schüler",
        "seite", "welche", "welcher", "wann", "wo", "wer", "zeige",
        "gib", "hilf", "beispiel", "definition", "bedeutet",
        "beschreibe", "nennen", "und", "oder", "nicht", "auch",
        "eine", "einen", "einem", "einer", "das", "die", "der",
        "den", "dem", "des", "ein", "kein", "keine", "mit", "von",
        "auf", "für", "über", "unter", "zwischen", "nach", "bei",
        "aus", "an", "in", "zu", "durch"
    ]
    text_lower = text.lower()
    words = text_lower.split()
    german_count = sum(1 for word in words if word in german_words)
    if german_count >= 1:
        return "german"
    return "english"


def detect_question_type(text):
    """Detect what type of answer the teacher needs"""
    text_lower = text.lower()

    if any(word in text_lower for word in [
        "how would i teach", "how to teach", "how do i teach",
        "teaching strategy", "how should i teach", "how can i teach",
        "tell students", "teach students", "strategy",
        "explain to students", "teach this", "teach me",
        "wie lehre ich", "wie erkläre ich", "wie kann ich unterrichten",
        "wie soll ich", "wie bringe ich", "unterrichten",
        "schülern erklären", "schülern beibringen"
    ]):
        return "teaching_strategy"

    elif any(word in text_lower for word in [
        "which page", "what page", "where is", "page number",
        "which line", "where can i find",
        "welche seite", "auf welcher seite", "wo steht",
        "wo ist", "welche zeile"
    ]):
        return "find_location"

    elif any(word in text_lower for word in [
        "give me example", "show me example", "example of",
        "gib mir beispiel", "zeige beispiel", "beispiel für",
        "give example", "show example"
    ]):
        return "examples"

    elif any(word in text_lower for word in [
        "what is", "define", "explain", "what are", "describe",
        "was ist", "was sind", "erkläre", "definiere", "beschreibe",
        "can u explain", "can you explain", "tell me what"
    ]):
        return "explanation"

    else:
        return "general"


def build_prompt(question_type, language, context_with_pages,
                 question, not_found_msg):
    """Build the correct prompt based on question type AND language"""

    # ── ENGLISH PROMPTS ──────────────────────────────────────────
    if language == "english":

        if question_type == "teaching_strategy":
            return f"""YOU MUST RESPOND IN ENGLISH ONLY. DO NOT USE ANY GERMAN WORDS.

You are an expert STEM teacher trainer helping teachers plan lessons.
Based ONLY on the following context from uploaded documents.

Use this format:

TEACHING STRATEGY: [concept name] (Page X)

Step 1 - Start with: [how to introduce the concept simply]
Step 2 - Explain: [the core concept in simple words]
Step 3 - Give this example: [best example to use with students]
Step 4 - Check understanding: [question to ask students]
Teacher tip: [one practical classroom tip]
---

If information not found say: "{not_found_msg}"

Context:
{context_with_pages}

Teacher question: {question}

Teaching advice in ENGLISH:"""

        elif question_type == "find_location":
            return f"""YOU MUST RESPOND IN ENGLISH ONLY. DO NOT USE ANY GERMAN WORDS.

You are a helpful STEM assistant.
Find where the topic is mentioned in the uploaded documents.
Based ONLY on the following context.

Use this format:

Found: [topic name]
Location: Page [X] of [filename]
Context: [exact description of where it appears]
---

If not found say: "{not_found_msg}"

Context:
{context_with_pages}

Question: {question}

Location answer in ENGLISH:"""

        elif question_type == "examples":
            return f"""YOU MUST RESPOND IN ENGLISH ONLY. DO NOT USE ANY GERMAN WORDS.

You are a helpful STEM assistant.
Provide clear examples based ONLY on the context.

Use this format:

Examples: [concept name] (Page X)
Example 1: [first example]
Example 2: [second example]
Real world use: [practical application]
---

If not found say: "{not_found_msg}"

Context:
{context_with_pages}

Question: {question}

Examples in ENGLISH:"""

        elif question_type == "explanation":
            return f"""YOU MUST RESPOND IN ENGLISH ONLY. DO NOT USE ANY GERMAN WORDS.

You are a helpful STEM assistant.
Explain the concept clearly and simply.
Based ONLY on the following context.

Use this format:

Concept: [name] (Page X)
Definition: [simple one sentence definition]
Why it matters: [one sentence]
Example: [one simple example]
---

If not found say: "{not_found_msg}"

Context:
{context_with_pages}

Question: {question}

Answer in ENGLISH:"""

        else:
            return f"""YOU MUST RESPOND IN ENGLISH ONLY. DO NOT USE ANY GERMAN WORDS.

You are a helpful STEM assistant.
Answer the question clearly and helpfully.
Based ONLY on the following context.
Always mention page numbers.

If not found say: "{not_found_msg}"

Context:
{context_with_pages}

Question: {question}

Answer in ENGLISH:"""

    # ── GERMAN PROMPTS ───────────────────────────────────────────
    else:

        if question_type == "teaching_strategy":
            return f"""DU MUSST AUSSCHLIESSLICH AUF DEUTSCH ANTWORTEN. KEIN ENGLISCH.

Du bist ein erfahrener MINT-Lehrertrainer.
Basiere deine Antwort NUR auf dem folgenden Kontext.

Verwende dieses Format:

UNTERRICHTSSTRATEGIE: [Konzeptname] (Seite X)

Schritt 1 - Einstieg: [Wie das Konzept einfach eingeführt wird]
Schritt 2 - Erklärung: [Das Kernkonzept in einfachen Worten]
Schritt 3 - Beispiel: [Bestes Beispiel für Schüler]
Schritt 4 - Verständniskontrolle: [Frage die man Schülern stellt]
Lehrertipp: [Ein praktischer Unterrichtstipp]
---

Wenn nicht gefunden: "{not_found_msg}"

Kontext:
{context_with_pages}

Lehrerfrage: {question}

Unterrichtsrat auf DEUTSCH:"""

        elif question_type == "find_location":
            return f"""DU MUSST AUSSCHLIESSLICH AUF DEUTSCH ANTWORTEN. KEIN ENGLISCH.

Du bist ein hilfreicher MINT-Assistent.
Finde wo das Thema im Dokument erwähnt wird.
Basiere deine Antwort NUR auf dem Kontext.

Verwende dieses Format:

Gefunden: [Themenname]
Ort: Seite [X] von [Dateiname]
Kontext: [Genaue Beschreibung wo es erscheint]
---

Wenn nicht gefunden: "{not_found_msg}"

Kontext:
{context_with_pages}

Frage: {question}

Antwort auf DEUTSCH:"""

        elif question_type == "examples":
            return f"""DU MUSST AUSSCHLIESSLICH AUF DEUTSCH ANTWORTEN. KEIN ENGLISCH.

Du bist ein hilfreicher MINT-Assistent.
Gib klare Beispiele basierend NUR auf dem Kontext.

Verwende dieses Format:

Beispiele: [Konzeptname] (Seite X)
Beispiel 1: [Erstes Beispiel]
Beispiel 2: [Zweites Beispiel]
Praxisanwendung: [Praktische Anwendung]
---

Wenn nicht gefunden: "{not_found_msg}"

Kontext:
{context_with_pages}

Frage: {question}

Antwort auf DEUTSCH:"""

        elif question_type == "explanation":
            return f"""DU MUSST AUSSCHLIESSLICH AUF DEUTSCH ANTWORTEN. KEIN ENGLISCH.

Du bist ein hilfreicher MINT-Assistent.
Erkläre das Konzept klar und einfach.
Basiere deine Antwort NUR auf dem Kontext.

Verwende dieses Format:

Konzept: [Name] (Seite X)
Definition: [Einfache Ein-Satz-Definition]
Warum wichtig: [Ein Satz]
Beispiel: [Ein einfaches Beispiel]
---

Wenn nicht gefunden: "{not_found_msg}"

Kontext:
{context_with_pages}

Frage: {question}

Antwort auf DEUTSCH:"""

        else:
            return f"""DU MUSST AUSSCHLIESSLICH AUF DEUTSCH ANTWORTEN. KEIN ENGLISCH.

Du bist ein hilfreicher MINT-Assistent.
Beantworte die Frage klar und hilfreich.
Basiere deine Antwort NUR auf dem Kontext.
Erwähne immer die Seitenzahl.

Wenn nicht gefunden: "{not_found_msg}"

Kontext:
{context_with_pages}

Frage: {question}

Antwort auf DEUTSCH:"""


def index_pdf(pdf_path, subject="General"):
    """Read PDF, split into chunks, ADD to existing ChromaDB"""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["subject"] = subject

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=get_embeddings()
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
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
    try:
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=get_embeddings()
        )

        llm = Ollama(model="mistral")

        if subject_filter and subject_filter != "All":
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 2,
                    "filter": {"subject": subject_filter}
                }
            )
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 2}
            )

        docs = retriever.invoke(question)

        if not docs:
            return "I could not find any relevant information in the uploaded documents.", []

        # Build context with page numbers
        context_with_pages = ""
        for doc in docs:
            page = doc.metadata.get("page", 0) + 1
            source = os.path.basename(
                doc.metadata.get("source", "Unknown")
            )
            context_with_pages += (
                f"[Source: {source}, Page {page}]\n"
                f"{doc.page_content}\n\n"
            )

        # Detect language and question type
        language = detect_language(question)
        question_type = detect_question_type(question)

        # Not found messages
        not_found_msg = (
            "Ich konnte diese Information nicht finden."
            if language == "german"
            else "I could not find this in the uploaded documents."
        )

        # Build the correct prompt
        prompt = build_prompt(
            question_type,
            language,
            context_with_pages,
            question,
            not_found_msg
        )

        # Get answer from Mistral
        answer = llm.invoke(prompt)

        # Build sources list
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0)
            subject = doc.metadata.get("subject", "General")
            sources.append(
                f"{os.path.basename(source)} "
                f"(page {page + 1}) [{subject}]"
            )

        sources = list(set(sources))

        return answer, sources

    except Exception as e:
        print(f"Error in ask_question: {e}")
        return "An error occurred. Please try again.", []