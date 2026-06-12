from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import re
from pypdf import PdfReader
from llm_client import LLMClient

app = FastAPI()

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Use absolute path for Jinja2
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def clean_transcript(text):
    """Membersihkan noise transkrip YouTube."""
    text = re.sub(r'\[\d{1,2}:\d{2}(:\d{2})?\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_llm_prompt(cleaned_text):
    """Membuat prompt untuk LLM agar menghasilkan KB dalam Bahasa Indonesia."""
    prompt = (
        "You are a professional knowledge engineer. I have a raw transcript "
        "that is repetitive and unstructured. Please transform it into a highly structured "
        "Knowledge Base for a RAG (Retrieval-Augmented Generation) system.\n\n"
        "IMPORTANT: The entire output MUST be in Indonesian language (Bahasa Indonesia).\n\n"
        "Guidelines:\n"
        "1. Use Markdown format.\n"
        "2. Extract key entities (e.g., products, fruits, medicines, symptoms) and translate them into Indonesian if they are common terms.\n"
        "3. Create a clear relation: Entity -> Mechanism/Reason -> Effect -> Related Items (all in Indonesian).\n"
        "4. Create a 'Panduan Umum' (General Guidelines) section for rules that apply to all entities.\n"
        "5. Remove all fillers, greetings, and repetitions.\n"
        "6. Keep the technical terms (e.g., drug classes, chemical names) but explain them in Indonesian if possible.\n\n"
        "Raw Transcript:\n"
        f"'''\n{cleaned_text}\n'''\n\n"
        "Structured Knowledge Base (in Indonesian):"
    )
    return prompt

#def generate_sft_prompt(cleaned_text):
#    """Membuat prompt untuk LLM agar menghasilkan dataset SFT (Instruction-Response pairs) dalam format JSONL."""
#    prompt = (
#        "You are an expert dataset curator for LLM Fine-Tuning. "
#        "I have a raw transcript that I want to convert into a high-quality Supervised Fine-Tuning (SFT) dataset.\n\n"
#        "Your goal is to create multiple 'Instruction-Response' pairs based on the content.\n\n"
#        "IMPORTANT: Both Instruction and Response MUST be in Indonesian (Bahasa Indonesia).\n\n"
#        "Guidelines:\n"
#        "1. Format: Output MUST be in JSONL (JSON Lines) format. Each line must be a valid JSON object.\n"
#        "2. Schema: Each line should follow this structure: {\"instruction\": \"...\", \"output\": \"...\"}\n"
#        "3. Diversity: Create a variety of instructions (e.g., direct questions, 'jelaskan proses...', 'apa yang terjadi jika...').\n"
#        "4. Accuracy: Ensure the responses are grounded strictly in the provided transcript.\n"
#        "5. Natural Language: Instructions should sound like real users, and responses should be helpful, complete, and professional.\n"
#        "6. Next-Token Prediction: Ensure the 'output' is a high-quality gold answer that a model should predict token by token.\n\n"
#        "Raw Transcript:\n"
#        f"'''\n{cleaned_text}\n'''\n\n"
#        "SFT Dataset in JSONL format (Indonesian):"
#    )
#    return prompt

def generate_sft_prompt(cleaned_text):
    """Membuat prompt untuk LLM agar menghasilkan dataset SFT dalam format JSON (Gemma 4 Best Practices)."""
    prompt = (
        "You are an expert dataset curator for LLM Fine-Tuning. "
        "I have a raw transcript that I want to convert into a high-quality Supervised Fine-Tuning (SFT) dataset "
        "following the Gemma 4 formatting standards.\\n\\n"
        "Your goal is to create multiple Chat interaction samples based on the content.\\n\\n"
        "IMPORTANT: Both User and Assistant contents MUST be in Indonesian (Bahasa Indonesia).\\n\\n"
        "Guidelines:\\n"
        "1. Format: Output MUST be a valid JSON array of objects. Each object represents one conversation.\\n"
        "2. Schema: Strictly follow this structure for each conversation object:\\n"
        '{\n  "messages": [\n    {"role": "system", "content": [{"type": "text", "text": "Kamu adalah asisten AI medis profesional yang bertindak sebagai konsultan kesehatan dan obat. Berikan informasi yang akurat secara ilmiah, gunakan bahasa yang mudah dipahami pasien, dan selalu ingatkan pasien untuk berkonsultasi dengan dokter untuk diagnosis final."}]},\n    {"role": "user", "content": [{"type": "text", "text": "TEXT_PERTANYAAN_PASIEN"}]},\n    {"role": "assistant", "content": [{"type": "text", "text": "TEXT_JAWABAN_MEDIS"}]}\n  ]\n}\n'
        "3. Content Format: The 'content' field MUST be a list of objects with 'type' and 'text' (e.g., [{\"type\": \"text\", \"text\": \"...\"}]).\\n"
        "4. Diversity: Create a variety of questions for the 'user' role (e.g., patient complaints, direct questions about drugs, 'apa efek samping jika...').\\n"
        "5. Accuracy: Ensure the assistant responses are grounded strictly in the provided transcript.\\n"
        "6. Natural Language: User content should sound like real, natural questions from ordinary people, and assistant responses should be empathetic, helpful, complete, and professional.\\n"
        "7. Strict JSON: Do not include any markdown code blocks (like ```json) or conversational text. Output ONLY the raw JSON array.\\n\\n"
        "Raw Transcript:\\n"
        f"'''\\n{cleaned_text}\\n'''\\n\\n"
        "SFT Dataset in JSON format (Gemma 4 style):"
    )
    return prompt


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/process", response_class=HTMLResponse)
async def process_pdf(
    request: Request, 
    file: UploadFile = File(...), 
    provider: str = Form(None), 
    model: str = Form(None), 
    api_key: str = Form(None),
    format_type: str = Form("kb")
):
    print(f"DEBUG: Received request to process file: {file.filename} with format: {format_type}")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        print(f"DEBUG: Attempting to save file to {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"DEBUG: File saved successfully to {file_path}")
        
        reader = PdfReader(file_path)
        raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        if not raw_text.strip():
            print("DEBUG: Extracted text is empty")
            raise ValueError("PDF contains no extractable text (it might be scanned/image-based)")
            
        print(f"DEBUG: Extracted {len(raw_text)} characters")
        cleaned = clean_transcript(raw_text)
        
        # Pemilihan Prompt berdasarkan format_type
        if format_type == "sft":
            prompt = generate_sft_prompt(cleaned)
        else:
            prompt = generate_llm_prompt(cleaned)
        
        # Integrasi LLM
        result = None
        if provider and model:
            try:
                print(f"DEBUG: Calling LLM {provider} with model {model}...")
                llm = LLMClient(provider, model, api_key)
                result = await llm.generate(prompt)
                print("DEBUG: LLM generation successful")
            except Exception as e:
                print(f"DEBUG LLM ERROR: {str(e)}")
                result = f"LLM Error: {str(e)}"

        return templates.TemplateResponse(request=request, name="index.html", context={
            "prompt": prompt,
            "result": result,
            "filename": file.filename,
            "format_type": format_type
        })
    except Exception as e:
        print(f"ERROR: Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(request=request, name="index.html", context={
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
