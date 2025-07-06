import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# point this at your local .gguf
MODEL_PATH = "mistral-7b-openorca.Q4_0.gguf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_chain(pdf_path: str):
    # load & split
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # embed & vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    vectordb = Chroma.from_documents(chunks, embeddings)

    # local Llama
    llm = LlamaCpp(
        streaming=False,
        model_path=MODEL_PATH,
        temperature=0.7,
        top_p=1,
        n_ctx=4096
    )

    # chain with memory (optional)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )

def ask_chain(chain, question: str) -> str:
    return chain.run({"question": question})
