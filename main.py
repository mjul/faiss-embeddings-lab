import pathlib
import PyPDF2

DATA_DIR = pathlib.Path(__file__).parent / "data"
DOCUMENT_DIR = DATA_DIR / "docs"


def extract_text_from_pdf(file):
    """
    Extract text from a PDF file.
    :param file: the path to the file
    :return: the text
    """
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def main():
    pdf_files = [str(p) for p in DOCUMENT_DIR.iterdir()
                 if p.suffix == ".pdf" and p.is_file()]

    text = extract_text_from_pdf(pdf_files[0])
    print(text)


if __name__ == "__main__":
    main()
