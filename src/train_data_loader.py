import pandas as pd
import torch.utils.data
import torch
import os
from transformers import BertTokenizer, BertModel, BertForMaskedLM


# NQDataset class
class NQDataset(torch.utils.data.Dataset):
    def __init__(self, path):

        self.df = pd.read_json(path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):

        # WE ARE NOT RANDOM SAMPLING HERE: we grab the Gold passage, the question, and the hardest negative passage.
        # Potential Ablation: sample hard negative randomly, sample positive passage randomly

        # 0 indexes us into the gold passage
        pos_ctx = self.df['positive_ctxs'][index][0]
        # [CLS] <title> [SEP] <text> [SEP]
        pos_ctx_token = self.tokenizer.encode(pos_ctx['title'], text_pair=pos_ctx['text'],
                                              add_special_tokens=True, max_length=256,
                                              padding='max_length', truncation=True)

        # 0 indexes us into the *hardest* hard negative passage (hard in terms of BM25)

        # if no hard negative contexts found, choose a regular negative one
        if len(self.df['hard_negative_ctxs'][index]) != 0:
            hard_neg_ctx = self.df['hard_negative_ctxs'][index][0]
        else:
            hard_neg_ctx = self.df['negative_ctxs'][index][0]

        # [CLS] <title> [SEP] <text> [SEP]
        hard_neg_ctx_token = self.tokenizer.encode(hard_neg_ctx['title'], text_pair=hard_neg_ctx['text'],
                                                   add_special_tokens=True, max_length=256,
                                                   padding='max_length', truncation=True)


        ques = self.df['question'][index]
        # [CLS] <question> [SEP]
        ques_token = self.tokenizer.encode(ques, add_special_tokens=True, max_length=64, padding='max_length', truncation=True)

        # 0 indexes us into the *first* answer in the answers list
        # We don't ever need this unless we're doing stuff with the reader, but we'll leave in
        answer = self.df['answers'][index][0]

        return torch.Tensor(ques_token), torch.Tensor(pos_ctx_token), torch.Tensor(hard_neg_ctx_token)

"""
train_set = NQDataset('./data/retriever/nq-dev.json')
loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)

for x,y,z in loader:
    print(x.shape, "\n")
    print(y.shape, "\n")
    print(z.shape, "\n")
    break
"""
