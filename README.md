# Implementation of LSA, LDA, and SBERT for Semantic Search

This project employs Latent Semantic Analysis (LSA), Latent Dirichlet Allocation (LDA), and Sentence-BERT (SBERT) on the MS MARCO dataset, enabling semantic search functionality across these models. By utilizing GloVe embeddings of the documents and comparing them with provided queries using cosine similarity, it establishes a baseline for model comparison. Key evaluation metrics such as Precision, Average Precision, Recall, F1-Score, and Mean Average Precision (MAP) are computed to assess model performance.

## Dependencies
    -- nltk
    -- tqdm
    -- gensim
    -- scipy
    -- numpy
    -- sklearn
    -- sentence_transformers
    -- Pytorch
    -- GloVe embeddings 
    
## Run Locally

Clone the project

```bash
    git clone https://github.com/zthsk/semantic_search.git
```

Go to the project directory

```bash
    cd semantic_search
```

Install dependencies

```bash
    pip install nltk
    pip install tqdm
    pip install gensim
    pip install scipy   
    pip install numpy
    pip install scikit-learn
    pip install sentence-transformers
    pip install torch torchvision torchaudio

```

Train the LSA, LDA, BERT, and GloVe
```bash
    python train_models.py --bert sbert_embeddings.npy
    python train_models.py --lsa lsa_model.pny
    python train_models.py --lda lda_model.pny
    python train_models.py --glove glove_embeddings.npy

```

Query the model with a single query
``` bash
    python query.py --model [bert, lsa, lda] --query "your query"
```
Query the model with a list of queries
``` bash
    ./run_queries.sh  # just update the queries you want in queries.txt
```



## Results of a query with different models 

![App Screenshot](https://github.com/zthsk/semantic_search/blob/main/sbert.png "result of bert model")

![App Screenshot](https://github.com/zthsk/semantic_search/blob/main/lsa.png "result of lsa model")

![App Screenshot](https://github.com/zthsk/semantic_search/blob/main/lda.png "result of lda model")



