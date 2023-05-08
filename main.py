import pathlib

DATA_DIR = pathlib.Path(__file__).parent / "data"
DOCUMENT_DIR = DATA_DIR / "docs"


def main():
    pdf_files = [str(p) for p in DOCUMENT_DIR.iterdir()
                 if p.suffix == ".pdf" and p.is_file()]

    print(pdf_files)




if __name__ == "__main__":
    main()
