import os
import re
import uuid
import json
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, jsonify
from flask_cors import CORS

from rag.rag_engine import RAGEngine
from rag import pdf_extractor, openrouter

# ── Load .env ──────────────────────────────────────────────────────
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if openrouter.is_available(OPENROUTER_KEY):
    print(f"[App] OpenRouter: ENABLED (key configured)")
else:
    print("[App] OpenRouter: OFFLINE (no key — using local engine only)")

# ── Flask setup ────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.secret_key = "nexaai_ultra_secret_2026"
app.config["UPLOAD_FOLDER"] = "uploads"
SESSIONS_META_DIR = os.path.join("data", "sessions")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(SESSIONS_META_DIR, exist_ok=True)

# ── Single global RAG engine (Lazy Initialized) ────────────────────
_rag_instance = None
def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance


# ── Session meta (persist filename across server restarts) ─────────
def _meta_path(sid):
    return os.path.join(SESSIONS_META_DIR, f"{sid}_meta.json")

def save_session_meta(sid, filename, text):
    try:
        with open(_meta_path(sid), "w", encoding="utf-8") as f:
            json.dump({"filename": filename, "preview": text[:2000]}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[App] Meta save warn: {e}")

def load_session_meta(sid):
    p = _meta_path(sid)
    if not os.path.exists(p): return {}
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def get_session_id():
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex
    return session["session_id"]


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/")
def home():
    sid  = get_session_id()
    meta = load_session_meta(sid)
    return render_template("index.html", pdf_filename=meta.get("filename", ""))


@app.route("/new-session", methods=["POST"])
def new_session():
    session["session_id"] = uuid.uuid4().hex
    session.modified = True
    return jsonify({"ok": True})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    sid     = get_session_id()
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    file    = request.files.get("pdf_file")

    if not file or not file.filename:
        msg = "Please select a valid PDF file."
        return jsonify({"ok": False, "error": msg}) if is_ajax else render_template(
            "index.html", message=msg, message_type="error",
            pdf_filename=load_session_meta(sid).get("filename", ""))

    try:
        original_name = file.filename
        safe_path     = os.path.join(app.config["UPLOAD_FOLDER"], f"{sid}_{original_name}")
        file.save(safe_path)

        # Smart extraction: digital text → PyMuPDF → Vision OCR
        text, meta = pdf_extractor.extract(safe_path, api_key=OPENROUTER_KEY)

        # Build RAG index
        success = get_rag().load_data(sid, text, filename=original_name)
        if not success:
            raise ValueError("Failed to build search index from document.")

        save_session_meta(sid, original_name, text)

        # Build info message
        info_parts = [f'"{original_name}" indexed ({meta["pages"]} pages)']
        if meta["image_pages"]:
            info_parts.append(f'{meta["image_pages"]} image page(s) OCR\'d')
        if meta.get("warnings"):
            info_parts.extend(meta["warnings"][:2])

        if is_ajax:
            return jsonify({
                "ok":       True,
                "filename": original_name,
                "pages":    meta["pages"],
                "ocr":      meta["image_pages"] > 0,
                "method":   meta["method"],
            })

        return render_template("index.html",
            message=" — ".join(info_parts),
            message_type="success",
            pdf_filename=original_name)

    except Exception as e:
        meta_saved = load_session_meta(sid)
        err = str(e)
        return (jsonify({"ok": False, "error": err}) if is_ajax
                else render_template("index.html", message=f"Error: {err}",
                                     message_type="error",
                                     pdf_filename=meta_saved.get("filename", "")))


@app.route("/ask", methods=["POST"])
def ask_question():
    sid     = get_session_id()
    is_ajax = request.is_json
    data    = request.get_json() if is_ajax else request.form
    question = (data.get("question") or "").strip() if data else ""

    if not question:
        msg = "Please enter a question."
        return (jsonify({"error": msg}), 400) if is_ajax else render_template(
            "index.html", message=msg, message_type="error",
            pdf_filename=load_session_meta(sid).get("filename", ""))

    meta = load_session_meta(sid)
    if not meta:
        msg = "Please upload a PDF document first."
        return (jsonify({"error": msg}), 400) if is_ajax else render_template(
            "index.html", message=msg, message_type="error", pdf_filename="")

    try:
        answer = get_rag().generate_answer(sid, question, openrouter_key=OPENROUTER_KEY)
    except Exception as e:
        answer = f"System error: {str(e)}"

    if is_ajax:
        return jsonify({"answer": answer})
    return render_template("index.html",
                           pdf_filename=meta.get("filename", ""))


if __name__ == '__main__':
    # Use the port provided by Railway/Render, or default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)