# FAISS Document Embeddings Lab

Taking document embeddings for a spin.

## Setup
- Install the `conda` environment from `environment.yml`
- Place documents to index in the `data/docs` folder.
- Run `main.py --extract-pdf-texts` to extract the text from the PDFs.
- Run `main.py --create-index` to build the FAISS index from the text files.

You are now ready to query the index.

- The script will download the transformer model to the `models` folder 
- The script indexes the documents and creates a `faiss.index` file in the `data` folder.

### Notes
It appears that the Windows version of `faiss-cpu` on the `pytorch` channel is not
working due to a missing `swigfaiss` DLL. However, the `faiss-cpu` package on the `conda-forge` 
channel works.

# Notebooks
There are a number of Jupyter notebooks in the `notebooks` folder with 
experiments. See [notebooks/README.md](notebooks/README.md) for details.
