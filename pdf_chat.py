import torch
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import threading

MODEL_PATH = "deepseek-r1-distill-llama-8b-q4_k_m.gguf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CONTEXT_CHUNKS = 4

chain_data_cache = None
cache_lock = threading.Lock()
model_lock = threading.Lock()

class SemanticChunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            keep_separator=True
        )
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['word_count'] = len(chunk.page_content.split())
            chunk.metadata['has_numbers'] = bool(re.search(r'\d+', chunk.page_content))
            chunk.metadata['has_quotes'] = '"' in chunk.page_content or "'" in chunk.page_content
        return chunks

class COTReasoner:
    def __init__(self, llm):
        self.llm = llm
        # Simplified prompt that works better with different models
        self.reasoning_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history", "is_follow_up", "is_vague_follow_up"],
            template="""Answer based ONLY on the document context and chat history provided.

Previous conversation:
{chat_history}

Document context:
{context}

Question: {question}

Instructions: Read the context carefully. If this is a follow-up ({is_follow_up}) or vague question ({is_vague_follow_up}), use previous conversation for context. Give a direct, clear answer.

Answer:"""
        )
        self.chain = self.reasoning_prompt | self.llm
    
    def reason(self, context: str, question: str, chat_history: str, is_follow_up: bool, is_vague_follow_up: bool) -> Tuple[str, str]:
        try:
            full_response = self.chain.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_history,
                "is_follow_up": is_follow_up,
                "is_vague_follow_up": is_vague_follow_up
            })
            
            if hasattr(full_response, 'content'):
                response_text = full_response.content
            else:
                response_text = str(full_response)
            
            answer = self.extract_final_answer(response_text)
            return response_text, answer
            
        except Exception as e:
            return f"Error: {str(e)}", "I encountered an error while processing your question."
    
    def extract_final_answer(self, response_text: str) -> str:
        """Enhanced answer extraction that works with multiple model formats"""
        
        print(f"DEBUG - Raw response: {response_text[:500]}")  # Debug log
        
        # First, extract content after </think> tag if it exists
        think_match = re.search(r'</think>\s*(.+)', response_text, re.DOTALL)
        if think_match:
            post_think_content = think_match.group(1).strip()
            print(f"DEBUG - Post-think content: {post_think_content[:200]}")
            if len(post_think_content) > 20:
                # Clean up the post-think content
                cleaned = re.sub(r'\*+', '', post_think_content).strip()
                # Take the main content, stopping at any meta-text
                main_content = re.split(r'\n(?=captivated by|⏱️|\d+\.\d+s)', cleaned)[0].strip()
                if len(main_content) > 20:
                    return main_content
        
        # Clean the response by removing think tags
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        cleaned_response = re.sub(r'\*+', '', cleaned_response).strip()
        
        print(f"DEBUG - Cleaned response: {cleaned_response[:200]}")
        
        # Pattern 1: Look for "Answer:" explicitly
        answer_match = re.search(r'Answer\s*[:\-]\s*(.+?)(?=\n\n|$)', cleaned_response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            answer = re.sub(r'\n+', ' ', answer).strip()
            if len(answer) > 10:
                return answer
        
        # Pattern 2: Look for summary/conclusion patterns
        final_patterns = [
            r'(?:So,?\s*)?(?:to summarize|in summary|the story)\s*[,:]?\s*(.+?)(?=\n|$)',
            r'(?:Therefore|Thus)[,:]?\s*(.+?)(?=\n|$)',
            r'The story.*?is about\s*(.+?)(?=\n|$)',
        ]
        
        for pattern in final_patterns:
            match = re.search(pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r'\n+', ' ', answer).strip()
                if len(answer) > 10:
                    return answer
        
        # Pattern 3: Take the first substantial complete sentence/paragraph
        paragraphs = [p.strip() for p in cleaned_response.split('\n') if p.strip()]
        for paragraph in paragraphs:
            if (len(paragraph) > 30 and 
                not paragraph.lower().startswith(('the user', 'from the context', 'based on', 'debug')) and
                not re.match(r'^\d+\.', paragraph.strip())):
                # Clean and return the paragraph
                clean_para = re.sub(r'\n+', ' ', paragraph).strip()
                return clean_para
        
        # Pattern 4: Extract the longest meaningful sentence
        sentences = re.split(r'[.!?]+', cleaned_response)
        valid_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 25 and 
                not sentence.lower().startswith(('the user', 'from the', 'based on', 'debug')) and
                not re.match(r'^\d+\.', sentence.strip())):
                valid_sentences.append(sentence)
        
        if valid_sentences:
            best_sentence = max(valid_sentences, key=len)
            return best_sentence.strip() + "."
        
        # Final fallback - return the cleaned response if it's substantial
        if len(cleaned_response) > 20:
            return cleaned_response[:300] + "..." if len(cleaned_response) > 300 else cleaned_response
        
        return "I can see information in the document, but I need a clearer question to provide a specific answer."

def build_chain(pdf_path: str) -> Dict[str, Any]:
    global chain_data_cache
    
    with cache_lock:
        if chain_data_cache is not None:
            print("Using cached chain data.")
            return chain_data_cache
    
    try:
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages")
        
        chunker = SemanticChunker()
        chunks = chunker.chunk_documents(docs)
        print(f"Created {len(chunks)} chunks")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device, "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
        )
        
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=None)
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": MAX_CONTEXT_CHUNKS, "fetch_k": 8}
        )
        
        # Optimized settings for better performance
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.1,  # Lower for more focused answers
            max_tokens=300,   # Reduced for more concise responses
            top_p=0.7,       # More focused sampling
            n_ctx=3072,      # Slightly reduced context
            n_batch=128,     # Smaller batches for stability
            verbose=False,   # Reduce noise
            use_mlock=True,
            n_threads=None,
            repeat_penalty=1.1  # Prevent repetition
        )
        
        reasoner = COTReasoner(llm)
        print("Chain built successfully!")
        
        chain_data = {
            'retriever': retriever,
            'reasoner': reasoner,
            'vectordb': vectordb,
            'chunks': chunks,
            'metadata': {
                'total_chunks': len(chunks),
                'total_pages': len(docs),
                'chunk_size': CHUNK_SIZE
            }
        }
        
        chain_data_cache = chain_data
        return chain_data
        
    except Exception as e:
        print(f"Error building chain: {str(e)}")
        raise e

def ask_chain(chain_data: Dict[str, Any], question: str, chat_history: List[Dict] = None, is_follow_up: bool = False, is_vague_follow_up: bool = False) -> Dict[str, str]:
    if chat_history is None:
        chat_history = []
    
    with model_lock:
        try:
            print(f"Processing question: {question}")
            
            history_text = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}" 
                for item in chat_history[-3:]
            ]) if chat_history else "No previous conversation."
            
            if is_vague_follow_up and chat_history:
                context = chat_history[-1].get('answer', '')
                context_chunks = 0
                print("Using previous answer as context for vague follow-up")
            else:
                enhanced_query = enhance_query(question, chat_history, is_follow_up)
                print(f"Enhanced query: {enhanced_query}")
                relevant_docs = chain_data['retriever'].invoke(enhanced_query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                context_chunks = len(relevant_docs)
                print(f"Retrieved {context_chunks} relevant documents")
                
                if context_chunks == 0 and chat_history:
                    context = chat_history[-1].get('answer', '') + "\n" + context
                    print("No relevant docs found; using prior answer as context")
            
            reasoning, answer = chain_data['reasoner'].reason(
                context=context[:1800],  # Reduced context size
                question=question,
                chat_history=history_text,
                is_follow_up=is_follow_up,
                is_vague_follow_up=is_vague_follow_up
            )
            
            print(f"Extracted answer: {answer}")
            
            return {
                'reasoning': reasoning,
                'answer': answer,
                'context_chunks': context_chunks
            }
            
        except Exception as e:
            print(f"Error in ask_chain: {str(e)}")
            return {
                'reasoning': f"Error during processing: {str(e)}",
                'answer': "I encountered an error while processing your question. Please try rephrasing it.",
                'context_chunks': 0
            }

def enhance_query(question: str, chat_history: List[Dict], is_follow_up: bool) -> str:
    try:
        if not chat_history:
            return question

        lower_q = question.lower().strip()
        if any(keyword in lower_q for keyword in ["yes", "provide", "sure", "okay", "more", "go on", "tell me"]) or len(lower_q) < 20:
            prev_question = chat_history[-1].get('question', '')
            return f"{prev_question} - {question} (follow-up for more details)"

        if is_follow_up:
            prev_question = chat_history[-1].get('question', '')
            enhanced = f"{question} (context from prior question: {prev_question})"
        else:
            enhanced = question

        recent_terms = set()
        for item in chat_history[-2:]:
            if 'question' in item:
                words = re.findall(r'\b\w{4,}\b', item['question'].lower())
                recent_terms.update(words)
        
        if recent_terms:
            context_terms = list(recent_terms)[:3]
            enhanced = f"{enhanced} (history terms: {' '.join(context_terms)})"
        
        return enhanced
        
    except Exception as e:
        print(f"Error enhancing query: {str(e)}")
        return question

def safe_chain_invoke(chain, inputs: Dict[str, Any]) -> str:
    try:
        result = chain.invoke(inputs)
        
        if hasattr(result, 'content'):
            return result.content
        elif isinstance(result, dict) and 'text' in result:
            return result['text']
        else:
            return str(result)
            
    except Exception as e:
        print(f"Chain invoke error: {str(e)}")
        return f"Error processing request: {str(e)}"

def validate_chain_data(chain_data: Dict[str, Any]) -> bool:
    required_keys = ['retriever', 'reasoner', 'vectordb', 'chunks', 'metadata']
    return all(key in chain_data for key in required_keys)

def test_chain(pdf_path: str) -> bool:
    try:
        chain_data = build_chain(pdf_path)
        result = ask_chain(chain_data, "What is this document about?")
        return 'answer' in result and result['answer'] != ""
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False