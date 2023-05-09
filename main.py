import datetime
import pathlib
import PyPDF2
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Define the device to use
# For now, we will use the CPU
device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use the device to move tensors to the GPU or CPU


ROOT_DIR = pathlib.Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENT_DIR = DATA_DIR / "docs"
INDEX_FILE = DATA_DIR / "index.faiss"

MODEL_DIR = ROOT_DIR / "models"

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=MODEL_DIR)
model = AutoModel.from_pretrained('bert-base-uncased')


def extract_text_from_pdf(file):
    """
    Extract text from a PDF file.
    :param file: the path to the file
    :return: the text
    """
    try:
        reader = PyPDF2.PdfReader(file)
    except PyPDF2.errors.PdfReadError:
        print(f"** Could not read PDF file: {file}  (PdfReadError)")
        return ""
    except PyPDF2.errors.DependencyError:
        print(f"** Could not read PDF file: {file}  (DependencyError)")
        return ""
    except Exception:
        print(f"** Could not read PDF file: {file}  (Exception)")
        return ""
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# Define a function to compute embeddings for a list of texts
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )

    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return model_output.last_hidden_state[:, 0, :].detach().cpu().numpy()


def parse_pdf_files(pdf_files):
    print("Parsing PDF files...")
    file_texts = []
    for i, file in enumerate(pdf_files):
        if i % 25 == 0:
            print(f"  Parsing file {i} of {len(pdf_files)}: {file}")
        file_texts.append(extract_text_from_pdf(file))
    return file_texts


def create_index(pdf_files):
    """Parse the PDF files and create the FAISS index."""
    file_texts = parse_pdf_files(pdf_files)

    print("Computing embeddings...")
    embeddings = get_embeddings(file_texts)
    print("Creating index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))
    print("Saving index...")
    faiss.write_index(index, str(INDEX_FILE))


def conditionally_create_or_update_index(pdf_files):
    """Create or update the index if any of the files are newer than the index."""
    newest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
    newest_pdf_timestamp = newest_pdf.stat().st_mtime
    # Rebuild index if necessary
    is_index_up_to_date = INDEX_FILE.exists() and INDEX_FILE.stat().st_mtime > newest_pdf_timestamp
    if is_index_up_to_date:
        print("Index is up-to-date.")
    else:
        create_index(pdf_files)


def main():
    print("Searching for PDF files...")
    pdf_files = [p for p in DOCUMENT_DIR.iterdir()
                 if p.suffix == ".pdf" and p.is_file()]

    pdf_files = pdf_files[:100] # TODO: remove this, for testing only

    conditionally_create_or_update_index(pdf_files)

    print("Loading index...")
    index = faiss.read_index(str(INDEX_FILE))

    print("Searching index...")
    # Retrieve a vector by ID from the FAISS index
    required_vector_id = 1
    vector = np.array([index.reconstruct(i)[required_vector_id] for i in range(index.ntotal)])
    print(vector)
    # D, I = index.search(xq, 5)

    print("Done.")


if __name__ == "__main__":
    main()
