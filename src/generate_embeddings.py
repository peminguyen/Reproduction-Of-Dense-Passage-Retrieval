import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from models import *
from train_data_loader import *
from evaluation_utils.py import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch_size", default="32")
    parser.add_argument("--model", help="path to pickled model")
    parser.add_argument("--wiki", help="path to wikipedia DB")
    args = parser.parse_args()

    # create wikipedia loader (it returns tokenized passages)
    wiki_set = WikiDataset(args.wiki)
    wiki_loader = torch.utils.data.DataLoader(wiki_set, batch_size=args.b)

    # create questions loader (it returns tokenized questions)

    model = torch.load(PATH)
    model.eval()
    with torch.no_grad():


if __name__ == '__main__':
    main()
