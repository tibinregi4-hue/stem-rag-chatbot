from flask import Flask, render_template, request, jsonify
from rag_engine import index_pdf, ask_question, get_uploaded_files
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"

@app.route("/")
def home():
    files = get_uploaded_files()
    return render_template("index.html", files=files)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    subject = request.form.get("subject", "General")
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    num_chunks = index_pdf(filepath, subject)
    files = get_uploaded_files()
    
    return jsonify({
        "message": f"✅ Successfully indexed {file.filename}",
        "chunks": num_chunks,
        "files": files
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    subject_filter = data.get("subject_filter", "All")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    answer, sources = ask_question(question, subject_filter)
    
    return jsonify({
        "answer": answer,
        "sources": sources
    })

@app.route("/files", methods=["GET"])
def get_files():
    files = get_uploaded_files()
    return jsonify({"files": files})

@app.route("/delete", methods=["POST"])
def delete_file():
    data = request.get_json()
    filename = data.get("filename", "")
    
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    files = get_uploaded_files()
    return jsonify({"message": f"Deleted {filename}", "files": files})

if __name__ == "__main__":
    app.run(debug=True, port=2004)