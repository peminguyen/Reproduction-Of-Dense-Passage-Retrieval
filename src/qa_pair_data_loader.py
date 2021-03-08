import pandas as pd
import torch.utils.data
import torch
import os
from ast import literal_eval

from transformers import BertTokenizer, BertModel, BertForMaskedLM

class QAPairDataset(torch.utils.data.Dataset):

    def __init__(self, path):

        self.df = pd.read_csv(path, sep='\t', names=['question', 'answer'], converters={'answer': eval})
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print(self.df.index)
        print(self.df)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        ques = self.df['question'][index]
        ques_token = self.tokenizer.encode(ques, add_special_tokens=True, max_length=64, padding='max_length',
                                           truncation=True)

        answer = self.df['answer'][index]

        return torch.Tensor(ques_token), answer



loader = QAPairDataset("head-test.csv")

for x, y in loader:
    print("\n\n\n")
    print(len(loader))
    print(x)
    print(y)
    break
