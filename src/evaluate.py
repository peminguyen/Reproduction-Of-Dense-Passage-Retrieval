import torch.utils.data
import torch
import os

from utils.evaluation_utils import *
from utils.wiki_data_loader import *
from utils.qa_pair_data_loader import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki", help="path to wikipedia DB")
    parser.add_argument("--qa_pair", help="path to qa pair csv")
    parser.add_argument("--world_size", help="world size")
    parser.add_argument("--v", help="experiment version")
    args = parser.parse_args()

    args.world_size = int(args.world_size)
    passage_embeddings = []
    question_embeddings = []
    for i in range(args.world_size):
        passage_embeddings.append(deserialize_vectors(f"./embeddings/{args.v}-psg-{i}.h5")))
        question_embeddings.append(deserialize_vectors(f"./embeddings/{args.v}-ques-{i}.h5"))))
    
    wiki_dataset = WikiDataset(wiki)
    qa_pair_dataset = QAPairDataset(qa_pair)

    ks = [20, 100]
    results = []
    for k_val in ks:
        result = evaluate_wiki(question_embeddings,
                               passage_embeddings,
                               wiki_dataset, qa_pair_dataset, k=k_val)
        results.append(result)
    
if __name__ == '__main__':
    main()
