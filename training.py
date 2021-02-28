#!/x0/arnavmd/python3/bin/python3
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
#from torch.nn.parallel import DataParallel

from utils import *
from data_loader import *
from model import *

#import wandb
#wandb.init()



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch size", default="10")
    parser.add_argument("--e", help="number of epochs", default="100")
    parser.add_argument("--lr", help="initial learning rate", default="1e-5")
    parser.add_argument("--v", help="experiment version", default="0.1")
    parser.add_argument("--train_set", help="path to training dataset (h5 file)", default='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-train.json')
    parser.add_argument("--dev_set", help="path to training dataset (h5 file)", default='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-dev.json')
    parser.add_argument("--m", help="additional comments", default="")
    parser.add_argument("--world_size", help="world size", default=4)
    args = parser.parse_args()

    LEARNING_RATE = float(args.lr) * float(args.world_size)
    EXPERIMENT_VERSION = args.v
    LOG_PATH = './logs/' + EXPERIMENT_VERSION + '/'

    os.environ['MASTER_ADDR'] = '127.0.0.1' #'10.57.23.164'
    os.environ['MASTER_PORT'] = '1234'
    #torch.manual_seed(0)    
    print(torch.cuda.is_available())
    
    mp.spawn(train, nprocs=int(args.world_size), args=(args,))

def train(gpu, args):

    torch.manual_seed(0)
    rank = gpu 
    torch.cuda.set_device(rank)
    print(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)


    train_set = NQDataset(args.train_set)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,num_replicas=args.world_size,rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=int(args.b), num_workers=0, pin_memory=True, sampler=train_sampler)

    dev_set = NQDataset(args.dev_set)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_set,num_replicas=args.world_size,rank=rank)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=int(args.b), num_workers=0, pin_memory=True, sampler=dev_sampler)
    
    net = BERT_QA().cuda(gpu) 
    model = nn.parallel.DistributedDataParallel(net, device_ids=[gpu], find_unused_parameters=True)
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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0, eps=1e-8)
    min_loss = 10
    train_log = os.path.join(LOG_PATH, 'train_log.txt')
    test_log = os.path.join(LOG_PATH, 'test_log.txt')

    print(rank)

    for epoch in range(int(args.e)):

        print("="*10 + "Epoch " + str(epoch) + "="*10)
        losses = []
        model.train()
        for batch_idx, (ques, pos_ctx, neg_ctx) in enumerate(train_loader):
           

            #if batch_idx > 50:
            #    break
         
            # TODO: clean this up (alternating positive/negative contexts)
            ques = ques.long().cuda(non_blocking=True)

            psg = torch.cat((pos_ctx, neg_ctx), dim=1)
            psg = psg.reshape((-1, 256)) 
            psg = psg.long().cuda(non_blocking=True)

            #net = net.cuda()

            optimizer.zero_grad()
            q_emb, p_emb = model(ques, psg)
            sim,idx = net.get_sim(q_emb, p_emb)
            loss = net.loss_fn(sim, idx)
            losses.append(loss.item())
            loss.mean().backward()
            optimizer.step()

            if batch_idx % log_interval == 0:

                #wandb.log({'train_loss': loss.item()})
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))

                with open(train_log, 'a+') as f:
                    f.write(str(loss.item()) + "\n")
            

        test_losses = []
        test_scores = []
        model.eval()        
        for batch_idx, (ques, pos_ctx, neg_ctx) in enumerate(dev_loader):
            #print(len(test_losses))
            #if batch_idx > 100:
            #    break
            with torch.no_grad():
                ques = ques.long().cuda(non_blocking=True)

                psg = torch.cat((pos_ctx, neg_ctx), dim=1)
                psg = psg.reshape((-1, 256)) 
                psg = psg.long().cuda(non_blocking=True)

                q_emb, p_emb = model(ques, psg)
                sim, idx = net.get_sim(q_emb.detach(), p_emb.detach())
                

                preds = np.argmax(sim.detach().cpu().numpy(), axis=1)
                score = np.sum(preds == idx.cpu().detach().numpy())/len(preds) 
                #print(np.sum(preds == idx.cpu().detach().numpy()))
                #score = top_k_accuracy_score(idx, sim, k=1)
                test_scores.append(score)

                loss = net.loss_fn(sim, idx)
                test_losses.append(loss.item())

        #wandb.log({'val_loss': np.mean(test_losses)})
        #wandb.log({'val_acc': np.mean(test_scores)})
        with open(test_log, 'a+') as f:
            f.write(str(np.mean(test_losses))) 

        print('Val Loss:', np.mean(test_losses), ' Val Acc:', np.mean(test_scores))


        
if __name__ == '__main__':
    #mp.freeze_support()
    main()
