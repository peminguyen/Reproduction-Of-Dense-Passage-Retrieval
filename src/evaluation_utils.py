import os
import h5py
import numpy as np
import pandas as pd
import faiss
import regex as re

# from DPR repo (Karpukhin et al, EMNLP 2020)
def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

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

# question_embeddings should be a list of matrices each of ~num_questions/4 x d
# passage_embeddings should be a list of matrices each of ~num_psgs/4 x d
def evaluate_wiki(question_embeddings, passage_embeddings, wiki_loader, qa_pair_loader, k=100):
    global_passage_embeddings = np.zeros((0, 100), dtype=np.float32)
    global_question_embeddings = np.zeros((0, 100), dtype=np.float32)

    print(passage_embeddings[0].shape)
    for passage_embedding in passage_embeddings:
        global_passage_embeddings = np.concatenate((global_passage_embeddings,
                                                    passage_embedding), axis=0)
    
    for question_embedding in question_embeddings:
        global_question_embeddings = np.concatenate((global_question_embeddings,
                                                  question_embedding), axis=0)
    
    print(type(global_passage_embeddings[0, 0]))
    # sort the passages and questions by the global indices
    print("sorting global passage embeddings")
    global_passage_embeddings = \
            global_passage_embeddings[np.argsort(global_passage_embeddings[:, 0])]
    print("sorting global question embeddings")
    global_question_embeddings = \
            global_question_embeddings[np.argsort(global_question_embeddings[:, 0])]
    print(type(global_passage_embeddings[0, 0]))

    # slice off the global indices because they will mess up FAISS
    global_passage_embeddings = np.ascontiguousarray(global_passage_embeddings[:, 1:])
    global_question_embeddings = np.ascontiguousarray(global_question_embeddings[:, 1:])
    print(type(global_passage_embeddings[0, 0]))

    index = faiss.IndexFlatIP(global_passage_embeddings.shape[1])
    index.add(global_passage_embeddings)

    # results is num_questions x k
    _, results = index.search(global_question_embeddings, k)
    
    correct = 0
    for i in range(results.shape[0]):

        # DataLoader.WikiDataset.df['passage'][results[i, :]] will be a list
        # of pandas series objects, so we index into [1] to get the actual entry
        # i.e. the passage text
        psg_texts = wiki_loader.dataset.df['passage'][results[i, :]][1]

        # these are the answers pertaining to the current question, indexed off
        # of the *global question index*
        answer_texts = qa_pair_loader.dataset.df['answers'][i]
        
        # normalize the passages
        normalized_psgs = [_normalize_answer(psg) for psg in psg_texts]

        # normalize the answers
        normalized_answers = [_normalize_answer(answer) for answer in answers_texts]

        # check to see if answer string in any of the passages
        if any([answer in passage for answer in normalized_answers for passage in normalized_passages]):
            correct += 1
    
    print(f"top-{k} accuracy is {correct / results.shape[0]}")
    return correct / results.shape[0]

# np.random.seed(1234)
# psgs = np.random.random((1000, 100)).astype('float32')
# questions = np.random.random((15, 100)).astype('float32')
# serialize_vectors("psgs.h5", psgs);
# serialize_vectors("questions.h5", questions);
# d_psgs = deserialize_vectors("psgs.h5");
# d_questions = deserialize_vectors("questions.h5")
# print(np.array_equal(psgs, d_psgs))
# print(np.array_equal(questions, d_questions))
# evaluate_wiki([questions], [psgs])

