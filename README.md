# ACM-BCB-2019-Tutorial

Slides, code, and examples for the Low-dimensional Representation of Biological Sequence Data tutorial at the ACM-BCB 2019 conference.

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

