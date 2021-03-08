import pandas as pd
import torch.utils.data
import torch
import os
from ast import literal_eval

from transformers import BertTokenizer, BertModel, BertForMaskedLM

MAGIC_DELIMITER = "#&%@"

class QAPairDataset(torch.utils.data.Dataset):

    def __init__(self, path):

        self.df = pd.read_csv(path, sep='\t', names=['question', 'answer'], converters={'answer': eval})
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        ques = self.df['question'][index]
        ques_token = self.tokenizer.encode(ques, add_special_tokens=True, max_length=64,
                                           padding='max_length', truncation=True)

        answer = self.df['answer'][index]
        # For some reason, the pytorch dataloader default_collate will 1. delete all items
        # in the list except for the first one, and 2. squish those items into a tuple and
        # then wrap into list. Hacky way to fix: add esoteric delimiter so that we can use
        # a dataloader still
        joined_answer = MAGIC_DELIMITER.join(answer)

        return torch.Tensor(ques_token), joined_answer

"""
dataset = QAPairDataset("head-test.csv")
loader = torch.utils.data.DataLoader(dataset, batch_size=2)

for x, y in loader:
    print(x)
    print(y)
"""
