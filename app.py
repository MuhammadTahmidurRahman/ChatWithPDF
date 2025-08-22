import asyncio
import os
import shutil
import time
import json
import re
from typing import List, Dict, Any
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pdf_chat import build_chain, ask_chain, validate_groq_connection, get_available_groq_models

app = FastAPI(title="Advanced PDF Chat with Groq API", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Enhanced cache and conversation memory
chain_cache: Dict[str, Dict] = {}
conversation_memory: Dict[str, List[Dict]] = {}
performance_metrics: Dict[str, List[float]] = {}

@app.on_event("startup")
async def startup_event():
    """Validate Groq API connection on startup"""
    if not validate_groq_connection():
        print("WARNING: Groq API connection failed. Please check your API key.")
    else:
        print("✅ Groq API connection successful!")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": get_available_groq_models(),
        "api_status": "connected" if validate_groq_connection() else "disconnected"
    })

@app.get("/bleu_status")
async def bleu_evaluation_status():
    """Get BLEU evaluation system status"""
    from bleu_evaluator import get_bleu_evaluator
    evaluator = get_bleu_evaluator()
    return evaluator.get_evaluation_summary()
    
@app.get("/api/status")
async def api_status():
    """Check API and system status"""
    groq_status = validate_groq_connection()
    return {
        "groq_api": "connected" if groq_status else "disconnected",
        "cached_pdfs": list(chain_cache.keys()),
        "active_conversations": len(conversation_memory),
        "available_models": get_available_groq_models()
    }

@app.post("/query")
async def query(
    request: Request,
    file: UploadFile = File(None),
    filename: str = Form(None),
    question: str = Form(...),
    stream: bool = Form(False)
):
    """Enhanced query handling with Groq API and streaming support"""
    
    # Validate API connection first
    if not validate_groq_connection():
        raise HTTPException(
            status_code=503, 
            detail="Groq API is not available. Please check your API key."
        )
    
    # Process file upload
    if file is not None:
        filename = file.filename
        path = os.path.join(UPLOAD_DIR, filename)
        
        try:
            with open(path, "wb") as out:
                shutil.copyfileobj(file.file, out)
            
            # Build chain with Groq API
            print(f"Building chain for {filename} with Groq API...")
            chain_data = build_chain(path)
            chain_cache[filename] = chain_data
            conversation_memory[filename] = []
            performance_metrics[filename] = []
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process PDF: {str(e)}"
            )
        
    elif filename in chain_cache:
        chain_data = chain_cache[filename]
    else:
        # Return JSON error response
        return JSONResponse({
            "filename": filename or "Unknown",
            "question": question,
            "answer": "No PDF loaded. Please upload a PDF first.",
            "reasoning": "Error: No PDF loaded",
            "response_time": "0.00s",
            "context_chunks": 0,
            "quality_score": 0,
            "bleu_score": result.get('bleu_metrics', {}).get('bleu_score', 0.0),  # Add this line
            "bleu_details": result.get('bleu_metrics', {}),  # Add this line
            "models_used": {
                "llm_model": "N/A",
                "embedding_model": "N/A", 
                "reasoning_engine": "N/A"
            }
        })
    
    # Get conversation history
    chat_history = conversation_memory.get(filename, [])
    
    # Enhanced follow-up detection
    is_follow_up = len(chat_history) > 0
    is_vague_follow_up = detect_vague_followup(question, chat_history)
    
    # Handle streaming requests
    if stream or request.headers.get("accept") == "text/event-stream":
        return StreamingResponse(
            stream_advanced_reasoning(
                chain_data, question, chat_history, filename, 
                is_follow_up, is_vague_follow_up
            ),
            media_type="text/event-stream"
        )
    
    # Regular processing with enhanced metrics
    start_time = time.perf_counter()
    
    try:
        result = ask_chain(
            chain_data, question, chat_history, 
            is_follow_up, is_vague_follow_up
        )
        elapsed = time.perf_counter() - start_time
        

        # Store performance metrics
        performance_metrics.setdefault(filename, []).append(elapsed)
        if len(performance_metrics[filename]) > 50:
            performance_metrics[filename] = performance_metrics[filename][-50:]
        
        # Add BLEU score extraction
        bleu_score = result.get('bleu_score', 0.0)
        bleu_metrics = result.get('bleu_metrics', {})

        # Enhanced conversation memory
        conversation_entry = {
            'question': question,
            'answer': result['answer'],
            'reasoning_length': len(result.get('reasoning', '')),
            'context_chunks': result.get('context_chunks', 0),
            'timestamp': time.time(),
            'response_time': elapsed,
            'is_follow_up': is_follow_up,
            'is_vague_follow_up': is_vague_follow_up,
            'models_used': result.get('models_used', {})
        }
        
        conversation_memory[filename].append(conversation_entry)
        
        # Keep conversation memory manageable
        if len(conversation_memory[filename]) > 15:
            conversation_memory[filename] = conversation_memory[filename][-15:]
        
        avg_response_time = sum(performance_metrics[filename]) / len(performance_metrics[filename])
        quality_score = calculate_quality_score(result)
        
        # Get models used information
        models_used = result.get('models_used', {})
        
        # Return enhanced JSON response with model information
        return JSONResponse({
            "filename": filename,
            "question": question,
            "answer": result['answer'],
            "reasoning": result['reasoning'],
            "context_chunks": result.get('context_chunks', 0),
            "response_time": f"{elapsed:.2f}s",
            "avg_response_time": f"{avg_response_time:.2f}s",
            "conversation_length": len(conversation_memory[filename]),
            "is_follow_up": is_follow_up,
            "quality_score": quality_score,
            "bleu_score": bleu_score,
            "models_used": {
                "llm_model": models_used.get('llm_model', 'llama-3.3-70b-versatile'),
                "embedding_model": models_used.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                "reasoning_engine": models_used.get('reasoning_engine', 'Advanced Chain-of-Thought with Groq API')
            }
        })
        
    except Exception as e:
        print(f"Query processing error: {str(e)}")
        # Return JSON error response with model info
        return JSONResponse({
            "filename": filename,
            "question": question,
            "answer": f"I encountered an error: {str(e)}. This might be due to API limits or connectivity issues.",
            "reasoning": f"Error details: {str(e)}",
            "response_time": "0.00s",
            "context_chunks": 0,
            "quality_score": 0,
            "bleu_score": 0.0,
            "models_used": {
                "llm_model": "Error",
                "embedding_model": "Error",
                "reasoning_engine": "Error"
            }
        }, status_code=500)

async def stream_advanced_reasoning(
    chain_data: Dict, question: str, chat_history: List[Dict], 
    filename: str, is_follow_up: bool, is_vague_follow_up: bool
):
    """Enhanced streaming with detailed reasoning steps and model information"""
    
    try:
        # Get models used information
        models_used = chain_data.get('metadata', {})
        
        # Step 1: Context Analysis
        llm_model = models_used.get('llm_model', 'Groq API')
        init_message = f'Initializing advanced reasoning with {llm_model}...'
        yield f"data: {json.dumps({'step': 'init', 'message': init_message, 'progress': 10})}\n\n"
        await asyncio.sleep(0.1)
        
        # Step 2: Model Information
        llm_model_name = models_used.get('llm_model', 'llama-3.3-70b-versatile')
        embedding_model_name = models_used.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        model_info = f"Using {llm_model_name} for reasoning and {embedding_model_name} for embeddings"
        yield f"data: {json.dumps({'step': 'models', 'message': model_info, 'progress': 15})}\n\n"
        await asyncio.sleep(0.1)
        
        # Step 3: Conversation Context
        if chat_history:
            context_msg = f"Analyzing {len(chat_history)} previous exchanges for context continuity"
            yield f"data: {json.dumps({'step': 'context', 'message': context_msg, 'progress': 25})}\n\n"
        else:
            yield f"data: {json.dumps({'step': 'context', 'message': 'Starting fresh conversation', 'progress': 25})}\n\n"
        await asyncio.sleep(0.1)
        
        # Step 4: Query Enhancement
        query_type = "vague follow-up" if is_vague_follow_up else "follow-up" if is_follow_up else "new question"
        query_msg = f'Processing {query_type}: enhancing search strategy'
        yield f"data: {json.dumps({'step': 'query', 'message': query_msg, 'progress': 40})}\n\n"
        await asyncio.sleep(0.1)
        
        # Step 5: Document Search
        search_msg = "Searching document with semantic similarity" if not is_vague_follow_up else "Using enhanced context from conversation history"
        yield f"data: {json.dumps({'step': 'search', 'message': search_msg, 'progress': 55})}\n\n"
        await asyncio.sleep(0.2)
        
        # Step 6: Reasoning Process
        reasoning_model = models_used.get('llm_model', 'Groq LLaMA-3.3-70B')
        reasoning_msg = f'Applying advanced chain-of-thought reasoning with {reasoning_model}...'
        yield f"data: {json.dumps({'step': 'reasoning', 'message': reasoning_msg, 'progress': 70})}\n\n"
        await asyncio.sleep(0.2)
        
        # Process the actual query
        start_time = time.perf_counter()
        result = ask_chain(chain_data, question, chat_history, is_follow_up, is_vague_follow_up)
        elapsed = time.perf_counter() - start_time
        
        # Step 7: Quality Assessment
        quality_score = calculate_quality_score(result)
        quality_msg = f'Answer quality score: {quality_score}/100'
        yield f"data: {json.dumps({'step': 'quality', 'message': quality_msg, 'progress': 85})}\n\n"
        await asyncio.sleep(0.1)
        
        # Step 8: Model Summary
        result_model = result.get('models_used', {}).get('llm_model', 'llama-3.3-70b-versatile')
        models_summary = f"Response generated using: {result_model}"
        yield f"data: {json.dumps({'step': 'models_summary', 'message': models_summary, 'progress': 95})}\n\n"
        await asyncio.sleep(0.1)
        
        # Step 9: Final Response
        final_data = {
            'step': 'answer', 
            'message': result['answer'], 
            'elapsed': f'{elapsed:.2f}s', 
            'progress': 100, 
            'models_used': result.get('models_used', {})
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        await asyncio.sleep(0.1)
        
        # Update conversation memory
        conversation_memory[filename].append({
            'question': question,
            'answer': result['answer'],
            'timestamp': time.time(),
            'response_time': elapsed,
            'quality_score': quality_score,
            'context_chunks': result.get('context_chunks', 0),
            'models_used': result.get('models_used', {})
        })
        
        complete_data = {
            'step': 'complete', 
            'message': 'Processing complete', 
            'reasoning': result.get('reasoning', '')[:500], 
            'models_used': result.get('models_used', {})
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        
    except Exception as e:
        error_data = {'step': 'error', 'message': f'Processing error: {str(e)}', 'progress': 0}
        yield f"data: {json.dumps(error_data)}\n\n"

def detect_vague_followup(question: str, chat_history: List[Dict]) -> bool:
    """Enhanced vague follow-up detection"""
    if not chat_history:
        return False
    
    lower_q = question.lower().strip()
    
    # Expanded vague indicators
    vague_patterns = [
        r'^(yes|yeah|ok|okay|sure)\b.*',
        r'^(more|tell me more|continue|go on)\b.*',
        r'^(what else|anything else|what about)\b.*',
        r'^(explain|elaborate|expand)\b.*',
        r'^(provide|give me|show me)\b.*',
        r'^(details|information|specifics)\b.*'
    ]
    
    # Length-based detection
    if len(lower_q) < 20:
        return True
    
    # Pattern-based detection
    for pattern in vague_patterns:
        if re.match(pattern, lower_q):
            return True
    
    return False

def calculate_quality_score(result: Dict[str, Any]) -> int:
    """Calculate answer quality score based on various factors"""
    score = 50  # Base score
    
    answer = result.get('answer', '')
    reasoning = result.get('reasoning', '')
    context_chunks = result.get('context_chunks', 0)
    
    # Length and detail score
    if len(answer) > 100:
        score += 15
    elif len(answer) > 50:
        score += 10
    
    # Context utilization score
    if context_chunks > 3:
        score += 15
    elif context_chunks > 0:
        score += 10
    
    # Reasoning depth score
    if len(reasoning) > 500:
        score += 10
    elif len(reasoning) > 200:
        score += 5
    
    # Specificity indicators
    if any(indicator in answer.lower() for indicator in ['specific', 'detailed', 'according to', 'the document states']):
        score += 10
    
    # Avoid generic responses penalty
    generic_phrases = ['i found information', 'the document mentions', 'it appears that']
    if any(phrase in answer.lower() for phrase in generic_phrases):
        score -= 5
    
    return min(max(score, 10), 100)  # Clamp between 10 and 100

@app.get("/conversation/{filename}")
async def get_conversation(filename: str):
    """Enhanced conversation history with analytics"""
    history = conversation_memory.get(filename, [])
    
    if not history:
        return {"conversation": [], "length": 0, "analytics": {}}
    
    # Calculate analytics
    response_times = [item.get('response_time', 0) for item in history if 'response_time' in item]
    quality_scores = [item.get('quality_score', 50) for item in history if 'quality_score' in item]
    
    analytics = {
        "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 50,
        "total_questions": len(history),
        "follow_up_ratio": sum(1 for item in history if item.get('is_follow_up', False)) / len(history) if history else 0,
        "models_used": history[-1].get('models_used', {}) if history else {}
    }
    
    return {
        "conversation": history[-10:],  # Return last 10 for display
        "length": len(history),
        "analytics": analytics
    }

@app.post("/clear_conversation")
async def clear_conversation(filename: str = Form(...)):
    """Clear conversation history and reset metrics"""
    if filename in conversation_memory:
        conversation_memory[filename] = []
    if filename in performance_metrics:
        performance_metrics[filename] = []
    
    return {"status": "cleared", "filename": filename}

@app.get("/analytics/{filename}")
async def get_analytics(filename: str):
    """Detailed analytics for a specific PDF conversation"""
    if filename not in conversation_memory:
        return {"error": "PDF not found"}
    
    history = conversation_memory.get(filename, [])
    metrics = performance_metrics.get(filename, [])
    
    if not history:
        return {"analytics": {"message": "No conversation data available"}}
    
    # Comprehensive analytics
    response_times = [item.get('response_time', 0) for item in history]
    quality_scores = [item.get('quality_score', 50) for item in history]
    context_chunks_used = [item.get('context_chunks', 0) for item in history]
    
    # Get models used from the latest conversation
    latest_models = history[-1].get('models_used', {}) if history else {}
    
    analytics = {
        "conversation_stats": {
            "total_exchanges": len(history),
            "follow_up_questions": sum(1 for item in history if item.get('is_follow_up', False)),
            "vague_follow_ups": sum(1 for item in history if item.get('is_vague_follow_up', False))
        },
        "performance_metrics": {
            "avg_response_time": round(sum(response_times) / len(response_times), 2) if response_times else 0,
            "fastest_response": round(min(response_times), 2) if response_times else 0,
            "slowest_response": round(max(response_times), 2) if response_times else 0,
            "total_processing_time": round(sum(response_times), 2)
        },
        "quality_metrics": {
            "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 50,
            "highest_quality": max(quality_scores) if quality_scores else 0,
            "lowest_quality": min(quality_scores) if quality_scores else 0,
            "avg_context_chunks": round(sum(context_chunks_used) / len(context_chunks_used), 1) if context_chunks_used else 0
        },
        "conversation_flow": {
            "engagement_ratio": round((len([item for item in history if item.get('is_follow_up', False)]) / len(history)) * 100, 1) if history else 0,
            "question_complexity": "High" if sum(len(item.get('question', '')) for item in history) / len(history) > 50 else "Medium"
        },
        "models_used": latest_models
    }
    
    return {"analytics": analytics, "filename": filename}

@app.post("/export_conversation")
async def export_conversation(filename: str = Form(...), format: str = Form("json")):
    """Export conversation history in various formats"""
    if filename not in conversation_memory:
        return {"error": "PDF not found"}
    
    history = conversation_memory.get(filename, [])
    
    if format.lower() == "json":
        return JSONResponse(content={
            "filename": filename,
            "exported_at": time.time(),
            "conversation": history,
            "analytics": (await get_analytics(filename))["analytics"]
        })
    
    # Could add other formats like CSV, TXT here
    return {"error": "Unsupported format"}

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    groq_status = validate_groq_connection()
    
    return {
        "status": "healthy" if groq_status else "degraded",
        "groq_api": "connected" if groq_status else "disconnected",
        "cached_pdfs": len(chain_cache),
        "active_conversations": len([k for k, v in conversation_memory.items() if v]),
        "total_questions_processed": sum(len(conv) for conv in conversation_memory.values()),
        "available_models": get_available_groq_models(),
        "system_info": {
            "upload_dir": UPLOAD_DIR,
            "upload_dir_exists": os.path.exists(UPLOAD_DIR)
        }
    }

@app.get("/debug/{filename}")
async def debug_info(filename: str):
    """Enhanced debug information"""
    debug_data = {"filename": filename, "found": False}
    
    if filename in chain_cache:
        chain_data = chain_cache[filename]
        debug_data.update({
            "found": True,
            "metadata": chain_data.get('metadata', {}),
            "conversation_length": len(conversation_memory.get(filename, [])),
            "chain_components": list(chain_data.keys()),
            "performance_history": performance_metrics.get(filename, [])[-10:],  # Last 10 response times
            "recent_questions": [
                item.get('question', '')[:100] + ('...' if len(item.get('question', '')) > 100 else '')
                for item in conversation_memory.get(filename, [])[-5:]
            ]
        })
    
    return debug_data

@app.post("/optimize_model")
async def optimize_model(
    filename: str = Form(...),
    target_metric: str = Form("speed")  # "speed" or "quality"
):
    """Dynamically optimize model selection based on usage patterns"""
    if filename not in conversation_memory:
        return {"error": "PDF not found"}
    
    history = conversation_memory[filename]
    
    if target_metric == "speed":
        # Recommend faster model for simple questions
        avg_question_length = sum(len(item.get('question', '')) for item in history) / len(history) if history else 0
        
        if avg_question_length < 30:
            recommended_model = "llama-3.1-8b-instant"
        else:
            recommended_model = "mixtral-8x7b-32768"
    else:  # quality
        # Keep the most capable model for complex reasoning
        recommended_model = "llama-3.3-70b-versatile"
    
    current_models = history[-1].get('models_used', {}) if history else {}
    
    return {
        "current_model": current_models.get('llm_model', 'unknown'),
        "recommended_model": recommended_model,
        "reason": f"Optimized for {target_metric} based on conversation patterns",
        "avg_question_length": sum(len(item.get('question', '')) for item in history) / len(history) if history else 0,
        "models_used": current_models
    }

@app.get("/models")
async def list_models():
    """List available Groq models with capabilities"""
    models_info = {
        "llama-3.3-70b-versatile": {
            "description": "Most capable model for complex reasoning and analysis",
            "best_for": "Complex questions, detailed analysis, multi-step reasoning",
            "speed": "Medium",
            "context_window": "128k tokens"
        },
        "llama-3.1-8b-instant": {
            "description": "Fast model for quick responses",
            "best_for": "Simple questions, quick lookups, basic summaries",
            "speed": "Very Fast", 
            "context_window": "128k tokens"
        },
        "mixtral-8x7b-32768": {
            "description": "Balanced model for general use",
            "best_for": "General questions, moderate complexity analysis",
            "speed": "Fast",
            "context_window": "32k tokens"
        },
        "gemma2-9b-it": {
            "description": "Alternative model with good instruction following",
            "best_for": "Instruction-based tasks, structured responses",
            "speed": "Fast",
            "context_window": "8k tokens"
        }
    }
    
    return {
        "available_models": models_info,
        "current_default": "llama-3.3-70b-versatile",
        "api_status": "connected" if validate_groq_connection() else "disconnected"
    }

# Additional middleware for better error handling
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )

# Startup message
@app.on_event("startup")
async def startup_message():
    print("🚀 Advanced PDF Chat with Groq API is starting...")
    print("📚 Features: Enhanced reasoning, conversation continuity, performance analytics")
    print("🤖 Default model: LLaMA-3.3-70B-Versatile for superior reasoning capabilities")
    
    if validate_groq_connection():
        print("✅ Groq API connection validated successfully!")
    else:
        print("⚠️ Warning: Groq API connection failed. Check your API key.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)