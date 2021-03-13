#!/x0/arnavmd/python3/bin/python3
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.utils import *
from utils.train_data_loader import *
from model import *
from utils.dist_utils import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch size", default="32")
    parser.add_argument("--e", help="number of epochs", default="100")
    parser.add_argument("--lr", help="initial learning rate", default="1e-5")
    parser.add_argument("--v", help="experiment version", default="0.1")
    parser.add_argument("--train_set", help="path to training dataset (h5 file)", default='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-train.json')
    parser.add_argument("--dev_set", help="path to training dataset (h5 file)", default='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-dev.json')
    parser.add_argument("--m", help="additional comments", default="")
    parser.add_argument("--world_size", help="world size", default=4)
    parser.add_argument("--model", help="DISTILBERT, ROBERTA, or BERT", default="BERT")
    parser.add_argument("--top_k", help="for the hard negative sampling ablation", default=1)
    parser.add_argument("--shuffle_seed", help="needed for shuffling from epoch to epoch", default=1179493354)
    args = parser.parse_args()

    LEARNING_RATE = float(args.lr) * float(args.world_size)
    EXPERIMENT_VERSION = args.v
    LOG_PATH = './logs/' + EXPERIMENT_VERSION + '/'

    os.environ['MASTER_ADDR'] = '127.0.0.1' #'10.57.23.164'
    os.environ['MASTER_PORT'] = '1234'
    print(torch.cuda.is_available())

    assert int(args.b) % int(args.world_size) == 0, "batch size must be divisible by world size"
    assert args.model == "DISTILBERT" or args.model == "ROBERTA" or args.model == "BERT"

    mp.spawn(train, nprocs=int(args.world_size), args=(args,))

def train(gpu, args):

    torch.manual_seed(0)
    rank = gpu
    torch.cuda.set_device(rank)
    print(rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=int(args.world_size),
                            rank=rank)


    train_set = NQDataset(args.train_set, k=int(args.top_k))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    num_replicas=int(args.world_size),
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=int(args.b)//int(args.world_size),
                                               num_workers=0, pin_memory=True,
                                               sampler=train_sampler)

    dev_set = NQDataset(args.dev_set, k=int(args.top_k))
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_set,
                                                                  num_replicas=int(args.world_size),
                                                                  rank=rank)
    dev_loader = torch.utils.data.DataLoader(dev_set,
                                             batch_size=int(args.b)//int(args.world_size),
                                             num_workers=0, pin_memory=True,
                                             sampler=dev_sampler)
    net = None
    if args.model == "DISTILBERT":
        net = DISTILBERT_QA().cuda(gpu)
    elif args.model == "ROBERTA":
        net = ROBERTA_QA().cuda(gpu)
    else:
        net = BERT_QA().cuda(gpu)

    model = nn.parallel.DistributedDataParallel(net,
                                                device_ids=[gpu], 
                                                find_unused_parameters=True)
    print("Downloaded models")

    LOG_PATH = './logs/' + args.v  + '/'
    LEARNING_RATE = float(args.lr)

    if os.path.exists(LOG_PATH):
        restore_latest(model, LOG_PATH)
    else:
        os.makedirs(LOG_PATH)

    with open(os.path.join(LOG_PATH, 'setup.txt'), 'a+') as f:
        f.write("\nVersion: " + args.v)
        f.write("\nBatch Size: " + args.b)
        f.write("\nInitial Learning Rate: " + args.lr)
        f.write("\nTraining Set: " + args.train_set)
        f.write("\nValidation Set: " + args.dev_set)
        f.write("\nComments: " + args.m)

    log_interval = 20
    max_score = -1
    train_log = os.path.join(LOG_PATH, 'train_log.txt')
    dev_log = os.path.join(LOG_PATH, 'dev_log.txt')

    print(rank)

    warmup_counter = 0
    list1 = []
    for epoch in range(int(args.e)):
        list1.append([])
        for batch_idx, (ques, pos_ctx, neg_ctx) in enumerate(train_loader):
            ques = ques.long().cuda(non_blocking=True)
            list1[epoch].append(ques.cpu())

        if epoch == 2:
            break

    print(list1[0][0] == list1[1][0])
    print(list1[1][0] == list1[2][0])
    print(list1[2][0])

if __name__ == '__main__':
    main()