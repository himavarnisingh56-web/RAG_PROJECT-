import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

RESUME_FOLDER = "resumes"
DB_FOLDER = "chroma_db"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    return splitter.split_text(text)


def main():
    all_chunks = []
    all_metadata = []

    for file_name in os.listdir(RESUME_FOLDER):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(RESUME_FOLDER, file_name)

            print(f"Processing: {file_name}")

            text = extract_text_from_pdf(pdf_path)
            chunks = split_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "resume_name": file_name,
                    "chunk_id": i
                })

    if not all_chunks:
        print("No resume chunks found.")
        return

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_texts(
        texts=all_chunks,
        embedding=embedding_model,
        metadatas=all_metadata,
        persist_directory=DB_FOLDER
    )

    print("All resumes stored successfully in ChromaDB.")
    print(f"Total chunks stored: {len(all_chunks)}")


if __name__ == "__main__":
    main()