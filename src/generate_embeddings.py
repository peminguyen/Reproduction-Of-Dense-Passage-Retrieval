import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from models import *
from train_data_loader import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch_size", default="32")
    parser.add_argument("--model", help="path to pickled model")
    args = parser.parse_args()


    model = torch.load(PATH)
    model.eval()
    with torch.no_grad():

if __name__ == '__main__':
    main()
