from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging
import warnings

logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

def extract_text_from_txt(file_path):
    with open(file_path,"r",encoding="utf-8") as file:
        return file.read()

def extract_text_from_pdf(file_path):
    raw_text = ''
    with open(file_path,"rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text+= text + "\n"
    return raw_text

def chunk_document_text(text,filename,chunk_size=1500,overlap=500):
    """Split text into smaller overlapping chunks with corrected separator hierarchy"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap = overlap,
        length_function = len,
        separators = ["\n\n","\n"," ",""]
    )
    chunks = text_splitter.split_text(text)
    documents=[]
    for chunk in chunks:
        doc=Document(page_content=chunk,metadata={"source":filename})
        documents.append(doc)
    return documents

def build_vector_database(chunks,database_folder="./tech_db",embeddings=None):
    if embeddings is None:
        print("Initializing Local Hugging Face Embedding Model...🧠")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(f"Storing {len(chunks)} chunks in techDB...")
    vector_db = Chroma.from_documents(
        documents = chunks,
        embedding=embeddings,
        persist_directory=database_folder
    )
    print("Database built and saved successfully")
    return vector_db