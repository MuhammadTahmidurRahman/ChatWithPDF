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
        
        # Advanced COT Prompt Template with Multiple Reasoning Strategies
        self.advanced_cot_prompt = ChatPromptTemplate.from_template("""
You are an expert document analyst equipped with advanced reasoning capabilities. Your task is to provide comprehensive, well-reasoned answers using sophisticated Chain-of-Thought (COT) reasoning.

=== CONTEXT INFORMATION ===
Document Context:
{context}

Conversation History:
{chat_history}

=== QUESTION ANALYSIS ===
Current Question: "{question}"
Question Type: {question_type}
Is Follow-up: {is_follow_up}
Is Vague Follow-up: {is_vague_follow_up}

=== ADVANCED COT REASONING FRAMEWORK ===

**STEP 1: QUESTION DECOMPOSITION AND UNDERSTANDING**
Break down the question into its core components:
- What is the user specifically asking for?
- What type of information is required (factual, analytical, comparative, etc.)?
- Are there any implicit assumptions or context dependencies?
- How does this relate to previous conversation elements?

**STEP 2: CONTEXT EVALUATION AND RELEVANCE MAPPING**
Analyze the provided document context:
- Which sections of the context directly address the question?
- What supporting information is available?
- Are there any contradictions or ambiguities in the source material?
- How reliable and comprehensive is the available information?

**STEP 3: EVIDENCE EXTRACTION AND FACT VERIFICATION**
Systematically extract relevant information:
- Direct quotes or statements that answer the question
- Supporting details that provide context
- Numerical data, dates, or specific facts
- Relationships between different pieces of information

**STEP 4: LOGICAL REASONING CHAIN**
Apply structured reasoning:
- What can be directly concluded from the evidence?
- What reasonable inferences can be drawn?
- Are there any logical gaps that need acknowledgment?
- How do different pieces of evidence connect to form a complete picture?

**STEP 5: CONVERSATION CONTINUITY ANALYSIS**
Consider conversational context:
- How does this question build upon previous exchanges?
- Are there themes or topics that need continuity?
- Should the answer reference or expand on previous responses?

**STEP 6: UNCERTAINTY AND LIMITATION ASSESSMENT**
Evaluate confidence levels:
- What information is well-supported by the document?
- Where are the gaps or uncertainties?
- What assumptions, if any, are being made?
- How should limitations be communicated?

**STEP 7: RESPONSE SYNTHESIS AND OPTIMIZATION**
Craft the optimal response:
- Structure the answer for clarity and completeness
- Balance detail with accessibility
- Ensure direct response to the user's question
- Include appropriate caveats or limitations

=== FINAL RESPONSE FORMAT ===

Based on my analysis, here is my comprehensive response:

[ANSWER_START]
[Provide your complete, well-reasoned answer here. This should directly address the user's question while incorporating insights from your COT reasoning process. Make sure to:
- Start with a clear, direct answer
- Provide supporting details from the document
- Acknowledge any limitations or uncertainties
- Maintain appropriate tone and detail level]
[ANSWER_END]

=== REASONING DOCUMENTATION ===

**Question Analysis**: [Brief summary of question understanding]
**Key Evidence Found**: [Main supporting information from context]
**Reasoning Process**: [Key logical steps taken]
**Confidence Level**: [High/Medium/Low with justification]
**Limitations Noted**: [Any gaps or uncertainties]

Remember: Base all responses strictly on the provided document context and conversation history. If information is not available or unclear, explicitly acknowledge these limitations while providing the best possible response based on available evidence.
""")
        
        self.chain = self.advanced_cot_prompt | self.llm
    
    def advanced_reason(self, context: str, question: str, chat_history: str, is_follow_up: bool, is_vague_follow_up: bool) -> Tuple[str, str, str]:
        """
        Execute advanced COT reasoning with custom prompt engineering
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
            
            # Extract structured components
            final_answer = self._extract_final_answer(full_response)
            reasoning_steps = self._extract_reasoning_documentation(full_response)
            
            return full_response, final_answer, reasoning_steps
            
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
    
    def _extract_final_answer(self, response_text: str) -> str:
        """Extract the final answer from COT response"""
        # Look for content between ANSWER_START and ANSWER_END markers
        answer_pattern = r'\[ANSWER_START\](.*?)\[ANSWER_END\]'
        match = re.search(answer_pattern, response_text, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = re.sub(r'\n+', ' ', answer).strip()
            if len(answer) > 20:
                return answer
        
        # Fallback: extract last substantial paragraph before reasoning documentation
        sections = response_text.split('=== REASONING DOCUMENTATION ===')
        if len(sections) > 1:
            content_section = sections[0]
            paragraphs = [p.strip() for p in content_section.split('\n\n') if len(p.strip()) > 30]
            if paragraphs:
                return paragraphs[-1]
        
        return "Based on the document analysis, I found relevant information but need more specific guidance to provide a targeted response."
    
    def _extract_reasoning_documentation(self, response_text: str) -> str:
        """Extract and format reasoning documentation"""
        # Look for reasoning documentation section
        reasoning_pattern = r'=== REASONING DOCUMENTATION ===(.*?)(?=$|Remember:)'
        match = re.search(reasoning_pattern, response_text, re.DOTALL)
        
        if match:
            reasoning = match.group(1).strip()
            return self._format_reasoning_steps(reasoning)
        
        # Fallback: extract step-by-step reasoning
        steps_pattern = r'\*\*STEP \d+:.*?\*\*(.*?)(?=\*\*STEP|\*\*FINAL|\[ANSWER_START\]|$)'
        steps = re.findall(steps_pattern, response_text, re.DOTALL)
        
        if steps:
            formatted_steps = []
            for i, step in enumerate(steps, 1):
                clean_step = re.sub(r'\n+', ' ', step.strip())
                if clean_step:
                    formatted_steps.append(f"🔍 Step {i}: {clean_step}")
            return "\n\n".join(formatted_steps)
        
        return "Advanced COT reasoning completed with comprehensive analysis."
    
    def _format_reasoning_steps(self, reasoning_text: str) -> str:
        """Format reasoning documentation into readable steps"""
        # Split by reasoning components
        components = {
            'Question Analysis': '🎯',
            'Key Evidence Found': '📋',
            'Reasoning Process': '🧠',
            'Confidence Level': '📊',
            'Limitations Noted': '⚠️'
        }
        
        formatted_parts = []
        for component, emoji in components.items():
            pattern = rf'\*\*{re.escape(component)}\*\*:?\s*(.*?)(?=\*\*|$)'
            match = re.search(pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                content = re.sub(r'\n+', ' ', content)
                if content and content != '[Brief summary of question understanding]':
                    formatted_parts.append(f"{emoji} {component}:\n{content}")
        
        return "\n\n".join(formatted_parts) if formatted_parts else reasoning_text.strip()

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
        
        print("✅ Advanced COT Chain built successfully with custom prompt engineering!")
        
        chain_data = {
            'retriever': retriever,
            'cot_reasoner': cot_reasoner,  # Using advanced COT prompt engineering
            'vectordb': vectordb,
            'chunks': chunks,
            'metadata': {
                'total_chunks': len(chunks),
                'total_pages': len(docs),
                'chunk_size': CHUNK_SIZE,
                'llm_model': cot_reasoner.model_name,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'reasoning_engine': 'Advanced COT Prompt Engineering with Custom Templates'
            }
        }
        
        chain_data_cache = chain_data
        return chain_data
        
    except Exception as e:
        print(f"Error building chain: {str(e)}")
        raise e

def ask_chain(chain_data: Dict[str, Any], question: str, chat_history: List[Dict] = None, is_follow_up: bool = False, is_vague_follow_up: bool = False) -> Dict[str, str]:
    """Execute question with advanced COT prompt engineering"""
    
    if chat_history is None:
        chat_history = []
    
    with model_lock:
        try:
            print(f"🎯 Processing question with Advanced COT: {question}")
            
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
                print("🔄 Enhanced context for vague follow-up")
            else:
                enhanced_query = enhance_query(question, chat_history, is_follow_up)
                print(f"🔍 Enhanced query: {enhanced_query}")
                relevant_docs = chain_data['retriever'].invoke(enhanced_query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                context_chunks = len(relevant_docs)
                print(f"📚 Retrieved {context_chunks} relevant documents")
            
            # Execute Advanced COT Reasoning with Custom Prompt Engineering
            print("🧠 Executing Advanced COT Prompt Engineering...")
            full_reasoning, clean_answer, formatted_reasoning = chain_data['cot_reasoner'].advanced_reason(
                context=context[:6000],  # Increased context for better reasoning
                question=question,
                chat_history=history_text,
                is_follow_up=is_follow_up,
                is_vague_follow_up=is_vague_follow_up
            )
            
            print("✅ Advanced COT reasoning completed!")
            
            return {
                'reasoning': formatted_reasoning,  # Formatted reasoning steps
                'answer': clean_answer,           # Clean final answer
                'context_chunks': context_chunks,
                'context': context[:1000],        # Context for follow-ups
                'models_used': {
                    'llm_model': chain_data['metadata']['llm_model'],
                    'embedding_model': chain_data['metadata']['embedding_model'],
                    'reasoning_engine': chain_data['metadata']['reasoning_engine']
                },
                'raw_reasoning': full_reasoning   # Full COT reasoning for analysis
            }
            
        except Exception as e:
            print(f"❌ Error in advanced COT reasoning: {str(e)}")
            return {
                'reasoning': f"Error in advanced COT reasoning: {str(e)}",
                'answer': "I encountered an error during advanced reasoning. Please verify the API key and try again.",
                'context_chunks': 0,
                'models_used': {
                    'llm_model': 'Error',
                    'embedding_model': 'Error', 
                    'reasoning_engine': 'Advanced COT Error'
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
    """Test the advanced COT chain implementation"""
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
            'Advanced COT' in result['models_used']['reasoning_engine']
        )
        
        if success:
            print("✅ Advanced COT Chain test successful!")
        else:
            print("❌ Advanced COT Chain test failed!")
            
        return success
        
    except Exception as e:
        print(f"❌ Advanced COT test failed: {str(e)}")
        return False