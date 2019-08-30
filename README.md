# ACM-BCB-2019-Tutorial

Slides, code, and examples for the Low-dimensional Representation of Biological Sequence Data tutorial at the ACM-BCB 2019 conference.

- word2vec-examples.ipynb: a Jupyter notebook with two simple examples using Word2Vec in the context of english words and biological sequence data.
- multilateration-examples.ipynb: a Jupyter notebook with two examples using multilateration to generate embeddings for DNA 3-mers and to generate and embedding for the Hamming graph over amino acid sequences of length 8.
- human_cds.txt: human coding sequences obtained from Ensembl Biomart 8/13/2019 (https://www.ensembl.org/biomart/martview/) Human genes (GRCh38.p12).
- sequences.txt: each sequence from humans_cds.txt on a separate line.
- parseData.py: generate sequences.txt given human_cds.txt.
- Multilateration
  - Code to generate resolving sets for Hamming graphs.
  - Embeddings of Hamming graphs and codons generated using multilateration.
- Word2Vec
  - Code to generate embeddings of k-mers based on the data in sequences.txt.
  - Embeddings of k-mers using Word2Vec
  - An NLP example of Word2Vec.
-TSNE
  - Code to visualize embeddings using t-SNE.

