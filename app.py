import asyncio
import os
import shutil
import time
import json
import re
from typing import List, Dict, Any
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pdf_chat import build_chain, ask_chain

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Cache and conversation memory
chain_cache: Dict[str, Dict] = {}
conversation_memory: Dict[str, List[Dict]] = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/query-stream")
async def query_stream_endpoint():
    """Handle the query-stream endpoint for JavaScript polling"""
    async def event_stream():
        yield "data: {\"message\": \"Stream endpoint available\"}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/query")
async def query(
    request: Request,
    file: UploadFile = File(None),
    filename: str = Form(None),
    question: str = Form(...)
):
    """Handle queries with context awareness and streaming reasoning"""
    
    # Process file upload
    if file is not None:
        filename = file.filename
        path = os.path.join(UPLOAD_DIR, filename)
        
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        
        # Build chain
        chain_data = build_chain(path)
        chain_cache[filename] = chain_data
        conversation_memory[filename] = []
        
    elif filename in chain_cache:
        chain_data = chain_cache[filename]
    else:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "filename": filename or "Unknown",
            "question": question,
            "answer": "No PDF loaded. Please upload a PDF first.",
            "reasoning": "Error: No PDF loaded",
            "elapsed": "0.00s"
        })
    
    # Get conversation history
    chat_history = conversation_memory.get(filename, [])
    
    # Detect follow-up: Any query in an active session is potentially context-related
    is_follow_up = len(chat_history) > 0  # Consider follow-up if history exists
    is_vague_follow_up = any(keyword in question.lower() for keyword in ["yes", "provide", "sure", "okay", "more", "go on", "tell me"]) or len(question.strip()) < 20
    
    start_time = time.perf_counter()
    
    # Check if streaming request
    if request.headers.get("accept") == "text/event-stream":
        return StreamingResponse(
            stream_reasoning(chain_data, question, chat_history, filename, is_follow_up, is_vague_follow_up),
            media_type="text/event-stream"
        )
    
    # Regular processing
    try:
        result = ask_chain(chain_data, question, chat_history, is_follow_up, is_vague_follow_up)
        elapsed = time.perf_counter() - start_time
        
        # Store in conversation memory
        conversation_memory[filename].append({
            'question': question,
            'answer': result['answer'],
            'timestamp': time.time()
        })
        
        # Keep only last 10 exchanges
        if len(conversation_memory[filename]) > 10:
            conversation_memory[filename] = conversation_memory[filename][-10:]
        
        response_data = {
            'filename': filename,
            'question': question,
            'answer': result['answer'],
            'reasoning': result['reasoning'],
            'context_chunks': result.get('context_chunks', 0),
            'elapsed': f"{elapsed:.2f}s",
            'conversation_length': len(conversation_memory[filename])
        }
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            **response_data
        })
        
    except Exception as e:
        print(f"Query processing error: {str(e)}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "filename": filename,
            "question": question,
            "answer": "I encountered an error while processing your question. Please try rephrasing it.",
            "reasoning": f"Error: {str(e)}",
            "elapsed": "0.00s"
        })

async def stream_reasoning(chain_data: Dict, question: str, chat_history: List[Dict], filename: str, is_follow_up: bool, is_vague_follow_up: bool):
    """Stream reasoning process to frontend"""
    
    try:
        yield f"data: {json.dumps({'step': 'context', 'message': 'Analyzing conversation context...'})}\n\n"
        await asyncio.sleep(0.1)
        
        if chat_history:
            yield f"data: {json.dumps({'step': 'context', 'message': f'Found {len(chat_history)} previous exchanges'})}\n\n"
        else:
            yield f"data: {json.dumps({'step': 'context', 'message': 'Starting fresh conversation'})}\n\n"
        
        yield f"data: {json.dumps({'step': 'search', 'message': 'Searching PDF content...' if not is_vague_follow_up else 'Using prior context for follow-up...'})}\n\n"
        await asyncio.sleep(0.1)
        
        yield f"data: {json.dumps({'step': 'reasoning', 'message': 'Applying chain-of-thought reasoning...'})}\n\n"
        await asyncio.sleep(0.2)
        
        start_time = time.perf_counter()
        result = ask_chain(chain_data, question, chat_history, is_follow_up, is_vague_follow_up)
        elapsed = time.perf_counter() - start_time
        
        yield f"data: {json.dumps({'step': 'thinking', 'message': result['reasoning'][:500]})}\n\n"
        await asyncio.sleep(0.1)
        
        yield f"data: {json.dumps({'step': 'answer', 'message': result['answer'], 'elapsed': f'{elapsed:.2f}s'})}\n\n"
        
        conversation_memory[filename].append({
            'question': question,
            'answer': result['answer'],
            'timestamp': time.time()
        })
        
        yield f"data: {json.dumps({'step': 'complete', 'message': 'Done'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'step': 'error', 'message': f'Error: {str(e)}'})}\n\n"

@app.get("/conversation/{filename}")
async def get_conversation(filename: str):
    """Get conversation history for a PDF"""
    history = conversation_memory.get(filename, [])
    return {"conversation": history, "length": len(history)}

@app.post("/clear_conversation")
async def clear_conversation(filename: str = Form(...)):
    """Clear conversation history"""
    if filename in conversation_memory:
        conversation_memory[filename] = []
    return {"status": "cleared", "filename": filename}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cached_pdfs": list(chain_cache.keys()),
        "active_conversations": len(conversation_memory)
    }

@app.get("/debug/{filename}")
async def debug_info(filename: str):
    """Debug information for a specific PDF"""
    if filename in chain_cache:
        chain_data = chain_cache[filename]
        return {
            "filename": filename,
            "metadata": chain_data.get('metadata', {}),
            "conversation_length": len(conversation_memory.get(filename, [])),
            "chain_components": list(chain_data.keys())
        }
    return {"error": "PDF not found in cache"}