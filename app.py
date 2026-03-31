from flask import Flask, render_template, request, jsonify
from rag_engine import index_pdf, ask_question
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400
    
    # Save PDF
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Index it
    num_chunks = index_pdf(filepath)
    
    return jsonify({
        "message": f"✅ Successfully indexed {file.filename}",
        "chunks": num_chunks
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    answer, sources = ask_question(question)
    
    return jsonify({
        "answer": answer,
        "sources": sources
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)