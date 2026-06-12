# PDF-RAG App for LLM Finetuning

Aplikasi RAG (Retrieval-Augmented Generation) yang dirancang untuk memproses dokumen PDF (terutama transkrip YouTube) menjadi Knowledge Base terstruktur untuk kebutuhan finetuning LLM, dengan fokus pada konteks lokal Indonesia.

## 🚀 Fitur Utama
- **PDF Processing**: Upload dan ekstraksi teks dari file PDF.
- **Knowledge Base Construction**: Mengolah data menjadi format yang siap digunakan untuk SFT (Supervised Fine-Tuning).
- **LLM Integration**: Menggunakan Gemma 4 untuk pengolahan data dan query.
- **Web Interface**: Interface simpel menggunakan FastAPI dan Jinja2 templates.

## 🛠️ Tech Stack
- **Backend**: Python 3.12, FastAPI
- **LLM**: Gemma 4
- **Frontend**: HTML, CSS (Static assets)

## 📦 Instalasi & Menjalankan Aplikasi

1. **Clone Repository**
   ```bash
   git clone https://github.com/USERNAME/REPO_NAME.git
   cd pdf-rag-app
   ```

2. **Setup Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Application**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

## 📂 Struktur Folder
- `main.py`: Entry point aplikasi dan definisi API.
- `llm_client.py`: Modul untuk komunikasi dengan LLM Gemma 4.
- `templates/`: File HTML untuk UI.
- `static/`: Asset CSS dan JavaScript.
- `uploads/`: Penyimpanan sementara file PDF yang diupload.
