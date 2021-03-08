import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist

from model import *
from wiki_data_loader import *
from qa_pair_data_loader import *
from evaluation_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch_size", default="32")
    parser.add_argument("--model", help="path to pickled model")
    parser.add_argument("--wiki", help="path to wikipedia DB")
    parser.add_argument("--qa_pair", help="path to qa pair csv")
    parser.add_argument("--world_size", help="world size")
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'

    print(torch.cuda.is_available())

    mp.spawn(create_embeddings, nprocs=int(args.world_size), args=(args,))

def create_embeddings(gpu, args):

    torch.manual_seed(0)
    rank = gpu
    torch.cuda.set_device(rank)
    print(rank)

    log_interval = 100

    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # create wikipedia loader (it returns tokenized passages)
    wiki_set = WikiDataset(args.wiki)
    wiki_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args.world_size,
                                                                   rank=rank)
    wiki_loader = torch.utils.data.DataLoader(wiki_set, batch_size=int(args.b), pin_memory=True,
                                              sampler=wiki_sampler)

    # create questions loader (it returns tokenized questions)
    qa_pair_set = QAPairDataset(args.qa_pair)
    qa_pair_sampler = torch.utils.data.distributed.DistributedSampler(qa_pair_set, num_replicas=args.world_size,
                                                                      rank=rank)
    qa_pair_loader = torch.utils.data.DataLoader(qa_pair_set, batch_size=int(args.b), pin_memory=True,
                                                 sampler=qa_pair_sampler)

    psg_embeddings = np.zeros((0, 768))
    ques_embeddings = np.zeros((0, 768))

    net = BERT_QA().cuda(gpu)
    net = torch.load(args.model)
    model = nn.parallel.DistributedDataParallel(net, device_ids=[gpu], find_unused_parameters=True)
    model.eval()

    print("==========embedding the passages==========")
    for batch_idx, (passage) in enumerate(wiki_loader):
        with torch.no_grad():
            passage = passage.long().cuda(non_blocking=True)

            _, p_emb = model(None, passage)

            p_emb = p_emb.detach().cpu().numpy()
            psg_embeddings = np.concatenate((psg_embeddings, p_emb), axis=0)

            if batch_idx % log_interval == 0:
                print(f'Embedded {batch_idx} passages')

    print("==========embedding the questions==========")
    for batch_idx, (ques, _) in enumerate(qa_pair_loader):
        with torch.no_grad():
            ques = ques.long().cuda(non_blocking=True)

            q_emb, _ = model(ques, None)
            ques_embeddings = np.concatenate((ques_embeddings, q_emb), axis=0)

            if batch_idx % log_interval == 0:
                print(f'Embedded {batch_idx} questions')


if __name__ == '__main__':
    main()
