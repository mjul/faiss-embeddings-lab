# FAISS Document Embeddings Lab

Taking document embeddings for a spin.

## Setup
- Install the `conda` environment from `environment.yml`
- Place documents to index in the `data/docs` folder.
- The script will download the transformer model to the `models` folder 
- The script indexes the documents and creates a `faiss.index` file in the `data` folder.

### Notes
It appears that the Windows version of `faiss-cpu` on the `pytorch` channel is not
working due to a missing `swigfaiss` DLL. However, the `faiss-cpu` package on the `conda-forge` 
channel works.
