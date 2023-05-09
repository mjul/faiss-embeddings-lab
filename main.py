import argparse
import datetime
import json
import pathlib
import PyPDF2
import PyPDF2.errors
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# ----------------------------------------------------------------
# Configure external files and directories
# ----------------------------------------------------------------

ROOT_DIR = pathlib.Path(__file__).parent

# This folder is for the input data
DATA_DIR = ROOT_DIR / "data"
# This is the main document directory with the PDFs
DOCUMENT_DIR = DATA_DIR / "docs"
# This is a tmp directory with the text from the PDFs.
# It can be rebuilt from the PDFs.
TEXT_DIR = DATA_DIR / "texts"

# The models directory is where we store the models, pre-trained and our index
# It can be rebuilt from the text files.
MODEL_DIR = ROOT_DIR / "models"

INDEX_FILE = MODEL_DIR / "index.faiss"
INDEX_METADATA_FILE = MODEL_DIR / "index.metadata.json"

# ----------------------------------------------------------------
# Configure the tokenizer and model
# ----------------------------------------------------------------

# Define the device to use, using a CUDA GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=MODEL_DIR)
model = AutoModel.from_pretrained('bert-base-uncased').to(device)

# ----------------------------------------------------------------

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


def find_pdfs():
    """Return a list of all PDF files in the DOCUMENT_DIR."""
    return [p for p in DOCUMENT_DIR.iterdir()
            if p.suffix == ".pdf" and p.is_file()]


def extract_pdf_texts():
    """Extract the text from the PDF files in the DOCUMENT_DIR and save them in the TEXT_DIR."""
    for f in find_pdfs():
        pdf_path = f.relative_to(DOCUMENT_DIR)
        text_file = TEXT_DIR / pdf_path.with_suffix(".txt")
        text_file.parent.mkdir(parents=True, exist_ok=True)

        if text_file.exists():
            print(f"Text already extracted from: {f}.")
        else:
            print(f"Extracting text from {f}...")
            text = extract_text_from_pdf(f)
            text_file.write_text(text, encoding="utf-8")

# ----------------------------------------------------------------

def find_texts():
    """Return a list of all text files in the TEXT_DIR."""
    return [p for p in TEXT_DIR.iterdir()
            if p.suffix == ".txt" and p.is_file()]


# Define a function to compute embeddings for a list of texts
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )

    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return model_output.last_hidden_state[:, 0, :].detach().cpu().numpy()


def create_index():
    """Create the FAISS index from the text files."""
    text_files = find_texts()

    print("Loading text files...")
    file_texts = [f.read_text(encoding="utf-8") for f in text_files]

    print("Computing embeddings...")
    embeddings = get_embeddings(file_texts)

    print("Creating index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))

    print("Saving index...")
    faiss.write_index(index, str(INDEX_FILE))
    print("Saving index metadata...")
    metadata = {
        "created": datetime.datetime.now().isoformat(),
        "files": [str(f.relative_to(TEXT_DIR)) for f in text_files],
    }
    INDEX_METADATA_FILE.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-pdf-texts", help="Extract the text from the PDF files.", action="store_true")
    parser.add_argument("--create-index", help="Create the index from the text files.", action="store_true")
    args = parser.parse_args()

    if args.extract_pdf_texts:
        extract_pdf_texts()

    if args.create_index:
        create_index()

    print("Loading index...")
    index = faiss.read_index(str(INDEX_FILE))
    index_metadata = json.loads(INDEX_METADATA_FILE.read_text(encoding="utf-8"))

    print("Searching index for a vector...")
    # Retrieve a vector by ID from the FAISS index
    required_vector_id = 1
    vector = np.array([index.reconstruct(i)[required_vector_id] for i in range(index.ntotal)])
    print(vector)

    print("Searching for matching embeddings...")
    qs = [
        "prolog and logic programming historical overview",
        "artificial intelligence with gpt",
        "object-oriented terminology and language design",
        "learning to summarize from human feedback",
        "eigenvalues, eigenvectors, and invariant subspaces in linear algebra",
        "financial networks and statistical physics",
        "lexers and lexer generators, dfas and context-free grammars"
    ]

    for q in qs:
        xq = get_embeddings([q])
        print("Searching for: ", q)
        D, I = index.search(xq, 5)
        print("Distances and indices: D, I", D, I)
        for i in I[0]:
            print(i, index_metadata['files'][i])
        print()

    print("Done.")


if __name__ == "__main__":
    main()
