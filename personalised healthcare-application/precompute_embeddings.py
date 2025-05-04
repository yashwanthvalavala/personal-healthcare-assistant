import logging
import torch
import pickle
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths for saving embeddings and texts
EMBEDDINGS_PATH = "embeddings.pkl"
TEXTS_PATH = "texts.pkl"

def load_pdf(file_path):
    try:
        start_time = time.time()
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded PDF with {len(documents)} pages in {time.time() - start_time:.2f} seconds")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise

def text_split(extracted_data):
    try:
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        chunks = text_splitter.split_documents(extracted_data)
        logger.info(f"Split into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise

def get_embeddings_model():
    try:
        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info(f"Hugging Face model loaded on {device} in {time.time() - start_time:.2f} seconds")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise

def encode_chunks(chunks, model):
    start_time = time.time()
    texts = [chunk.page_content for chunk in chunks]
    batch_size = 32  # Process in batches
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings)
    logger.info(f"Encoded {len(texts)} chunks in {time.time() - start_time:.2f} seconds")
    return texts, embeddings

def save_embeddings(embeddings, texts):
    try:
        start_time = time.time()
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)
        with open(TEXTS_PATH, "wb") as f:
            pickle.dump(texts, f)
        logger.info(f"Embeddings and texts saved in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        raise

def main():
    file_path = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
    start_time = time.time()
    model = get_embeddings_model()
    extracted_data = load_pdf(file_path)
    text_chunks = text_split(extracted_data)
    logger.info(f"Length of chunks: {len(text_chunks)}")
    texts, embeddings = encode_chunks(text_chunks, model)
    save_embeddings(embeddings, texts)
    logger.info(f"Total embedding generation time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()