# www.youtube.com/@PythonCodeCampOrg
""" Subscribe to PYTHON CODE CAMP or I'll eat all your cookies... """

import torch
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# pick GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load your PDF
loader = PyPDFLoader(file_path=r"D:\Download\Chat_with_PDF-main\Chat_with_PDF-main\Sachin.pdf")
data = loader.load()

# split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data)

# initialize Llama for answer generation (streaming disabled)
llm_answer_gen = LlamaCpp(
    model_path=r"D:\Download\Chat_with_PDF-main\Chat_with_PDF-main\mistral-7b-openorca.Q4_0.gguf",
    streaming=False,           # <-- turn off streaming
    temperature=0.75,
    top_p=1,
    f16_kv=True,
    verbose=False,
    n_ctx=4096,
)

# embeddings + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)
vector_store = Chroma.from_documents(text_chunks, embeddings)

# conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
answer_gen_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_answer_gen,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# interactive loop
while True:
    user_input = input("Enter a question (or 'q' to quit): ")
    if user_input.strip().lower() == 'q':
        break
    answer = answer_gen_chain.run({"question": user_input})
    print("\nAnswer:", answer, "\n")

