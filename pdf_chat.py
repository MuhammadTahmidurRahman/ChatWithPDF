import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optimized settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_CONTEXT_CHUNKS = 6

# Global cache and locks
chain_data_cache = None
cache_lock = threading.Lock()
model_lock = threading.Lock()

class SemanticChunker:
    """Enhanced chunking with semantic awareness"""
    
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

class AdvancedCOTPromptEngineering:
    """
    Advanced Chain of Thought Prompt Engineering System
    This class demonstrates sophisticated prompt engineering techniques for COT reasoning
    """
    
    def __init__(self):
        self.model_name = "llama-3.3-70b-versatile"
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=0.1,
            max_tokens=2000,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Advanced COT Prompt Template with Better Separation
        self.advanced_cot_prompt = ChatPromptTemplate.from_template("""
You are an expert document analyst. Your task is to provide comprehensive, well-reasoned answers using advanced Chain-of-Thought reasoning. You MUST provide your response in the exact format specified below.

=== DOCUMENT CONTEXT ===
{context}

=== CONVERSATION HISTORY ===
{chat_history}

=== CURRENT QUESTION ===
Question: "{question}"
Question Type: {question_type}
Is Follow-up: {is_follow_up}
Is Vague Follow-up: {is_vague_follow_up}

Now, analyze this question using advanced reasoning, then provide your response in the EXACT format below:

[FINAL_ANSWER_START]
[Provide your clear, direct answer to the user's question here. This should be focused, comprehensive, and directly address what the user is asking. Do not include reasoning steps or analysis here - just the clean, final answer.]
[FINAL_ANSWER_END]

[REASONING_ANALYSIS_START]
**🎯 Question Understanding:**
[Analyze what the user is specifically asking for and the context of their question]

**📋 Evidence Extraction:**
[List the key pieces of information from the document that support your answer]

**🧠 Logical Reasoning:**
[Explain the step-by-step reasoning process used to arrive at your conclusion]

**📊 Confidence Assessment:**
[Evaluate your confidence level in the answer and explain why]

**⚠️ Limitations & Uncertainties:**
[Acknowledge any gaps, assumptions, or areas where more information would be helpful]
[REASONING_ANALYSIS_END]

Remember: 
1. Keep the FINAL_ANSWER section clean and focused - no reasoning steps there
2. Put all analysis, reasoning, and methodology in the REASONING_ANALYSIS section
3. Base everything on the provided document context
4. Be clear about limitations and confidence levels
""")
        
        self.chain = self.advanced_cot_prompt | self.llm
    
    def advanced_reason(self, context: str, question: str, chat_history: str, is_follow_up: bool, is_vague_follow_up: bool) -> Tuple[str, str, str]:
        """
        Execute advanced COT reasoning with clear separation of answer and reasoning
        """
        try:
            # Determine question type for better prompt customization
            question_type = self._classify_question(question, is_follow_up, is_vague_follow_up)
            
            # Execute the advanced COT prompt
            response = self.chain.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_history,
                "is_follow_up": is_follow_up,
                "is_vague_follow_up": is_vague_follow_up,
                "question_type": question_type
            })
            
            full_response = response.content if hasattr(response, 'content') else str(response)
            
            # Extract components with improved parsing
            final_answer = self._extract_clean_answer(full_response)
            reasoning_analysis = self._extract_reasoning_analysis(full_response)
            
            return full_response, final_answer, reasoning_analysis
            
        except Exception as e:
            error_msg = f"Advanced COT reasoning encountered an error: {str(e)}"
            return f"Error: {str(e)}", error_msg, f"Error in reasoning process: {str(e)}"
    
    def _classify_question(self, question: str, is_follow_up: bool, is_vague_follow_up: bool) -> str:
        """Classify question type for optimal prompt engineering"""
        if is_vague_follow_up:
            return "Vague Follow-up (requires context expansion)"
        elif is_follow_up:
            return "Contextual Follow-up (builds on previous exchange)"
        
        # Analyze question patterns
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['what', 'who', 'when', 'where']):
            return "Factual Information Request"
        elif any(word in q_lower for word in ['why', 'how', 'explain']):
            return "Analytical/Explanatory Question"
        elif any(word in q_lower for word in ['compare', 'difference', 'similar', 'contrast']):
            return "Comparative Analysis"
        elif '?' not in question:
            return "Statement/Command (requires interpretation)"
        else:
            return "General Inquiry"
    
    def _extract_clean_answer(self, response_text: str) -> str:
        """Extract only the clean final answer, without any reasoning"""
        # Look for content between FINAL_ANSWER_START and FINAL_ANSWER_END markers
        answer_pattern = r'\[FINAL_ANSWER_START\](.*?)\[FINAL_ANSWER_END\]'
        match = re.search(answer_pattern, response_text, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            # Clean up the answer - remove extra whitespace but preserve structure
            answer = re.sub(r'\n{3,}', '\n\n', answer)  # Replace 3+ newlines with 2
            answer = re.sub(r'[ \t]+', ' ', answer)      # Replace multiple spaces/tabs with single space
            return answer.strip()
        
        # Fallback: try to extract a clean answer from the beginning
        lines = response_text.split('\n')
        potential_answer = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('=== ') and not line.startswith('**') and not line.startswith('['):
                if len(line) > 20:  # Only substantial lines
                    potential_answer.append(line)
                    if len(potential_answer) >= 3:  # Take first few substantial lines
                        break
        
        if potential_answer:
            return ' '.join(potential_answer)
        
        return "I found relevant information in the document, but need a more specific question to provide a targeted response."
    
    def _extract_reasoning_analysis(self, response_text: str) -> str:
        """Extract and format the reasoning analysis section"""
        # Look for reasoning analysis section
        reasoning_pattern = r'\[REASONING_ANALYSIS_START\](.*?)\[REASONING_ANALYSIS_END\]'
        match = re.search(reasoning_pattern, response_text, re.DOTALL)
        
        if match:
            reasoning = match.group(1).strip()
            return self._format_reasoning_display(reasoning)
        
        # Fallback: look for structured reasoning components
        components = self._extract_reasoning_components(response_text)
        if components:
            return self._format_reasoning_display('\n\n'.join(components))
        
        return "Advanced COT reasoning completed with comprehensive document analysis."
    
    def _extract_reasoning_components(self, text: str) -> List[str]:
        """Extract reasoning components from unstructured text"""
        components = []
        
        # Look for emoji-marked sections
        patterns = [
            (r'🎯[^🎯📋🧠📊⚠️]*', 'Question Analysis'),
            (r'📋[^🎯📋🧠📊⚠️]*', 'Evidence Found'),
            (r'🧠[^🎯📋🧠📊⚠️]*', 'Reasoning Process'),
            (r'📊[^🎯📋🧠📊⚠️]*', 'Confidence Level'),
            (r'⚠️[^🎯📋🧠📊⚠️]*', 'Limitations')
        ]
        
        for pattern, name in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                content = matches[0].strip()
                if len(content) > 10:  # Only if substantial content
                    components.append(content)
        
        return components
    
    def _format_reasoning_display(self, reasoning_text: str) -> str:
        """Format reasoning for better display in the UI"""
        if not reasoning_text:
            return "Reasoning analysis completed."
        
        # Split by major sections
        sections = []
        current_section = ""
        
        lines = reasoning_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith('**'):
                # New section header
                if current_section:
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section:
            sections.append(current_section.strip())
        
        # Format sections nicely
        formatted_sections = []
        for section in sections:
            if section and len(section.strip()) > 5:
                # Clean up the section
                section = re.sub(r'\n{2,}', '\n', section)  # Remove extra newlines
                formatted_sections.append(section)
        
        return '\n\n'.join(formatted_sections) if formatted_sections else reasoning_text

def build_chain(pdf_path: str) -> Dict[str, Any]:
    """Build enhanced RAG chain with advanced COT prompt engineering"""
    
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
        
        # Use optimized embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
        )
        
        vectordb = Chroma.from_documents(
            chunks, 
            embeddings,
            persist_directory=None
        )
        
        # Enhanced retriever
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": MAX_CONTEXT_CHUNKS, 
                "fetch_k": 12,
                "lambda_mult": 0.7
            }
        )
        
        # Initialize Advanced COT Prompt Engineering System
        cot_reasoner = AdvancedCOTPromptEngineering()
        
        print("✅ Advanced COT Chain built successfully with improved answer extraction!")
        
        chain_data = {
            'retriever': retriever,
            'cot_reasoner': cot_reasoner,
            'vectordb': vectordb,
            'chunks': chunks,
            'metadata': {
                'total_chunks': len(chunks),
                'total_pages': len(docs),
                'chunk_size': CHUNK_SIZE,
                'llm_model': cot_reasoner.model_name,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'reasoning_engine': 'Advanced COT with Separated Answer/Reasoning'
            }
        }
        
        chain_data_cache = chain_data
        return chain_data
        
    except Exception as e:
        print(f"Error building chain: {str(e)}")
        raise e

def ask_chain(chain_data: Dict[str, Any], question: str, chat_history: List[Dict] = None, is_follow_up: bool = False, is_vague_follow_up: bool = False) -> Dict[str, str]:
    """Execute question with improved answer/reasoning separation"""
    
    if chat_history is None:
        chat_history = []
    
    with model_lock:
        try:
            print(f"🎯 Processing question with Improved COT: {question}")
            
            # Build conversation history
            history_text = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}" 
                for item in chat_history[-4:]
            ]) if chat_history else "No previous conversation."
            
            # Retrieve relevant context
            if is_vague_follow_up and chat_history:
                # Enhanced context for vague follow-ups
                prev_context = chat_history[-1].get('context', '')
                enhanced_query = f"{chat_history[-1].get('question', '')} {question}"
                relevant_docs = chain_data['retriever'].invoke(enhanced_query)
                new_context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
                context = f"Previous context:\n{prev_context}\n\nAdditional context:\n{new_context}"
                context_chunks = len(relevant_docs)
                print("🔥 Enhanced context for vague follow-up")
            else:
                enhanced_query = enhance_query(question, chat_history, is_follow_up)
                print(f"🔍 Enhanced query: {enhanced_query}")
                relevant_docs = chain_data['retriever'].invoke(enhanced_query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                context_chunks = len(relevant_docs)
                print(f"📚 Retrieved {context_chunks} relevant documents")
            
            # Execute Advanced COT Reasoning with Improved Separation
            print("🧠 Executing Improved COT with Answer/Reasoning Separation...")
            full_reasoning, clean_answer, formatted_reasoning = chain_data['cot_reasoner'].advanced_reason(
                context=context[:6000],
                question=question,
                chat_history=history_text,
                is_follow_up=is_follow_up,
                is_vague_follow_up=is_vague_follow_up
            )
            
            print("✅ Improved COT reasoning completed with clean separation!")
            
            return {
                'reasoning': formatted_reasoning,  # Clean reasoning only
                'answer': clean_answer,           # Clean answer only
                'context_chunks': context_chunks,
                'context': context[:1000],
                'models_used': {
                    'llm_model': chain_data['metadata']['llm_model'],
                    'embedding_model': chain_data['metadata']['embedding_model'],
                    'reasoning_engine': chain_data['metadata']['reasoning_engine']
                },
                'raw_reasoning': full_reasoning[:2000]  # Truncated for performance
            }
            
        except Exception as e:
            print(f"❌ Error in improved COT reasoning: {str(e)}")
            return {
                'reasoning': f"Error in reasoning: {str(e)}",
                'answer': "I encountered an error during processing. Please verify the API key and try again.",
                'context_chunks': 0,
                'models_used': {
                    'llm_model': 'Error',
                    'embedding_model': 'Error', 
                    'reasoning_engine': 'Improved COT Error'
                }
            }

def enhance_query(question: str, chat_history: List[Dict], is_follow_up: bool) -> str:
    """Enhanced query building with context integration"""
    try:
        if not chat_history:
            return question

        lower_q = question.lower().strip()
        
        # Detect vague follow-ups
        vague_indicators = [
            "yes", "more", "continue", "go on", "tell me more", "elaborate", 
            "explain", "what else", "provide", "show me", "give me"
        ]
        
        if any(indicator in lower_q for indicator in vague_indicators) or len(lower_q) < 25:
            prev_question = chat_history[-1].get('question', '')
            return f"{prev_question} - provide more detailed information about: {question}"

        # Combine with previous context for follow-ups
        if is_follow_up and len(chat_history) > 0:
            prev_terms = []
            for item in chat_history[-2:]:
                if 'question' in item:
                    words = re.findall(r'\b[A-Z][a-z]+|\b[a-z]{4,}\b', item['question'])
                    prev_terms.extend(words[:3])
            
            if prev_terms:
                enhanced = f"{question} (related to: {' '.join(set(prev_terms)[:5])})"
            else:
                enhanced = question
        else:
            enhanced = question

        return enhanced
        
    except Exception as e:
        print(f"Error enhancing query: {str(e)}")
        return question

def validate_groq_connection() -> bool:
    """Validate Groq API connection"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY not found in environment variables")
            return False
        
        # Test connection
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=50,
            groq_api_key=api_key
        )
        
        response = llm.invoke("Hello, respond with 'Connection successful'")
        return "successful" in response.content.lower()
        
    except Exception as e:
        print(f"Groq connection test failed: {str(e)}")
        return False

def get_available_groq_models() -> List[str]:
    """Get list of available Groq models"""
    return [
        "llama-3.3-70b-versatile",       # Best for complex reasoning
        "llama-3.1-8b-instant",         # Fastest responses
        "gemma2-9b-it",                 # Good instruction following
        "mixtral-8x7b-32768"            # Balanced performance
    ]

def test_advanced_cot_chain(pdf_path: str) -> bool:
    """Test the improved COT chain implementation"""
    try:
        if not validate_groq_connection():
            print("❌ Groq API connection failed")
            return False
            
        chain_data = build_chain(pdf_path)
        result = ask_chain(chain_data, "What is this document about?")
        
        success = (
            'answer' in result and 
            result['answer'] != "" and
            'reasoning' in result and
            'Separated' in result['models_used']['reasoning_engine']
        )
        
        if success:
            print("✅ Improved COT Chain test successful!")
        else:
            print("❌ Improved COT Chain test failed!")
            
        return success
        
    except Exception as e:
        print(f"❌ Improved COT test failed: {str(e)}")
        return False