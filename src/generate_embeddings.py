import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from collections import OrderedDict
import glob


from model import *
from utils.wiki_data_loader import *
from utils.qa_pair_data_loader import *
from utils.evaluation_utils import *
from utils.utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch_size", default="32")
    parser.add_argument("--wiki", help="path to wikipedia DB")
    parser.add_argument("--qa_pair", help="path to qa pair csv")
    parser.add_argument("--world_size", help="world size")
    parser.add_argument("--v", help="experiment version")
    parser.add_argument("--model", help="DISTILBERT, ROBERTA, or BERT", default="BERT")
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'

    print(torch.cuda.is_available())

    assert int(args.b) % int(args.world_size) == 0, "batch size must be divisible by world size"
    assert args.model == "DISTILBERT" or args.model == "ROBERTA" or args.model == "BERT"

    mp.spawn(create_embeddings, nprocs=int(args.world_size), args=(args,))

def create_embeddings(gpu, args):

    args.world_size = int(args.world_size)
    torch.manual_seed(0)
    rank = gpu
    torch.cuda.set_device(rank)
    print(rank)

    log_interval = 100

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size, rank=rank)

    # create wikipedia loader (it returns tokenized passages)
    wiki_set = WikiDataset(args.wiki)
    wiki_sampler = torch.utils.data.distributed.DistributedSampler(wiki_set, 
                                                                   num_replicas=args.world_size,
                                                                   rank=rank)
    wiki_loader = torch.utils.data.DataLoader(wiki_set,
                                              batch_size=int(args.b)//int(args.world_size),
                                              pin_memory=True,
                                              sampler=wiki_sampler)

    # create questions loader (it returns tokenized questions)
    qa_pair_set = QAPairDataset(args.qa_pair)
    qa_pair_sampler = torch.utils.data.distributed.DistributedSampler(qa_pair_set,
                                                                      num_replicas=args.world_size,
                                                                      rank=rank)
    qa_pair_loader = torch.utils.data.DataLoader(qa_pair_set,
                                                 batch_size=int(args.b)//int(args.world_size),
                                                 pin_memory=True,
                                                 sampler=qa_pair_sampler)

    # concatenate the indices:
    # index 0: *global* index of the passage in the wikipedia database
    # index 1-768: the BERT embedding of the passage's CLS token
    psg_embeddings = np.zeros((0, 769), dtype=np.float32)

    # We do the same thing for the questions
    ques_embeddings = np.zeros((0, 769), dtype=np.float32)

    net = None
    if args.model == "DISTILBERT":
        net = DISTILBERT_QA().cuda(gpu)
    elif args.model == "ROBERTA":
        net = ROBERTA_QA().cuda(gpu)
    else:
        net = BERT_QA().cuda(gpu)

    # https://stackoverflow.com/a/44319982
    checkpoints = sorted(glob.glob('./logs/' + args.v  + '/' + '/*.pt'), key=os.path.getmtime)

    state_dict = torch.load(checkpoints[-1])
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:]
        new_dict[new_key] = v
    net.load_state_dict(new_dict)

    net = net.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(net,
                                                device_ids=[gpu],
                                                find_unused_parameters=True)
    model.eval()

    print("==========embedding the passages==========")
    for batch_idx, (passage, psg_indices) in enumerate(wiki_loader):
        with torch.no_grad():
            passage = passage.long().cuda(non_blocking=True)

            _, p_emb = model(None, passage)
            
            np_psg_indices = np.expand_dims(psg_indices.numpy(), axis=1)
            p_emb = p_emb.detach().cpu().numpy()
            p_emb = np.concatenate((np_psg_indices, p_emb), axis=1)
            psg_embeddings = np.concatenate((psg_embeddings, p_emb), axis=0)

            if batch_idx % log_interval == 0:
                print(f'Embedded {batch_idx} batches of passages')

    print("==========embedding the questions==========")
    for batch_idx, (ques, ques_indices) in enumerate(qa_pair_loader):
        with torch.no_grad():
            ques = ques.long().cuda(non_blocking=True)

            q_emb, _ = model(ques, None)

            np_ques_indices = np.expand_dims(ques_indices.numpy(), axis=1)

            q_emb = q_emb.detach().cpu().numpy()
            q_emb = np.concatenate((np_ques_indices, q_emb), axis=1)
            ques_embeddings = np.concatenate((ques_embeddings, q_emb), axis=0)

            if batch_idx % log_interval == 0:
                print(f'Embedded {batch_idx} batches of questions')

    serialize_vectors(psg_embeddings, f"./embeddings/{args.v}-psg-{rank}.h5")
    serialize_vectors(ques_embeddings, f"./embeddings/{args.v}-ques-{rank}.h5")

if __name__ == '__main__':
    main()
