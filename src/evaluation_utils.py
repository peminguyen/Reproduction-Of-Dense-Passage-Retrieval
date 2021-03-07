import os
import h5py
import numpy as np
import pandas as pd
import faiss

# matrix should be size n x d
def serialize_vectors(filename, matrix):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset("data", data=matrix)
    h5f.close()

# reads in h5 of numpy array and returns it
def deserialize_vectors(filename):
    h5f = h5py.File(filename, 'r')
    out = h5f["data"][:]
    return out

# queries should be num_queries x d
# embeddings should be num_psgs x d
def evaluate_wiki(query_embeddings, passage_embeddings, query_df, wiki_df, k=100):
    index = faiss.IndexFlatIP(passage_embeddings.shape[1])
    # print(np.max(np.linalg.norm(passage_embeddings, axis=1)))
    # print(np.argmax(np.linalg.norm(passage_embeddings, axis=1)))
    index.add(passage_embeddings)
    # results is num_queries x k
    _, results = index.search(query_embeddings, k)
    for i in range(query_embeddings.shape[0]):
        psg_texts = wiki_df['passage'][results[i, :]]
        answers = query_df['answers']
        # normalize psgs
        # normalize answers
        # is_correct = any([answer in passage for answer in normalized_answers for passage in normalized_passages])


np.random.seed(1234)
psgs = np.random.random((1000, 100)).astype('float32')
queries = np.random.random((15, 100)).astype('float32')
# psgs = np.arange(100).reshape(10, 10).astype('float32')
# norms_psgs = np.linalg.norm(psgs, axis=1)
# norms_queries = np.linalg.norm(queries, axis=1)
# psgs = psgs / np.expand_dims(norms_psgs, axis=1)
# queries = queries / np.expand_dims(norms_queries, axis=1)
serialize_vectors("psgs.h5", psgs);
serialize_vectors("queries.h5", queries);
d_psgs = deserialize_vectors("psgs.h5");
d_queries = deserialize_vectors("queries.h5")
print(np.array_equal(psgs, d_psgs))
print(np.array_equal(queries, d_queries))
evaluate_wiki(queries, psgs)
