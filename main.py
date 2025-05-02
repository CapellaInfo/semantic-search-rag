from fastapi import FastAPI, UploadFile, Form
import shutil
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

app = FastAPI()

UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstore"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Configuração da OpenAI
OPENAI_API_KEY = ""

# Função para carregar documentos
def load_document(file_path: str):
    ext = file_path.split(".")[-1].lower()
    
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        return None
    
    return loader.load()

# Endpoint para upload de documentos
@app.post("/upload/")
async def upload_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    documents = load_document(file_path)
    
    if not documents:
        return {"error": "Formato de arquivo não suportado"}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # chunk_size = split do texto em 1000
    # chunk_overlap = contexto do texto anterior = 200
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)

    return {"message": f"Arquivo {file.filename} processado com sucesso"}

# Endpoint para responder perguntas
@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

    chat_model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(chat_model, retriever=vectorstore.as_retriever())

    response = qa_chain.run(query)

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)