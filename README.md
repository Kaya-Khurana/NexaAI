# NexaAI: Universal Production RAG Engine v5

A high-performance, production-grade Universal RAG (Retrieval-Augmented Generation) engine designed to extract precise information from any document—digital or scanned—with zero domain-specific training.

## 🚀 Key Features
- **Hybrid Intelligence**: Combines a local zero-dependency extraction pipeline (BM25 + Structural Parsing) with a cloud-based reasoning fallback.
- **NVIDIA NIM Integration**: Uses **Llama 3.1 405B** for state-of-the-art Q&A and **Llama 3.2 90B Vision** for OCR and scanned document analysis.
- **Universal Extraction**: Specifically optimized for resumes, invoices, legal contracts, and technical manuals.
- **Production-Ready**: Supports session management, smart confidence scoring, and force-API routing for 100% precision.
- **Privacy First**: Fully supports offline-only mode (local core) or secured API integration via `.env`.

## 🛠 Tech Stack
- **Backend**: Python, Flask
- **Extraction**: `pdfplumber`, `PyMuPDF`
- **Intelligence**: NVIDIA NIM API (Llama 3.1/3.2)
- **Local RAG**: Pure Python BM25 + Multi-strategy structural parser

## 🚦 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file and add your NVIDIA API key:
   ```env
   OPENROUTER_API_KEY=nvapi-YOUR_KEY_HERE
   ```

3. **Run the App**:
   ```bash
   python app.py
   ```
   Access at `http://127.0.0.1:5000`

## 📄 License
MIT
