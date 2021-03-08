import pandas as pd
import torch.utils.data
import torch
import os

from transformers import BertTokenizer, BertModel, BertForMaskedLM

class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, path):

        self.df = pd.read_csv(path, sep='\t', names=['index', 'passage', 'title'])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def  __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        passage = self.df['passage'][index]
        title = self.df['title'][index]
        passage_token = self.tokenizer.encode(title, text_pair=passage,
                                              add_special_tokens=True, max_length=256,
                                              padding='max_length', truncation=True)

        return torch.Tensor(passage_token)





loader =  WikiDataset(r"test.tsv")
for x in loader:
    print(x.shape, "\n")
    print(x)

