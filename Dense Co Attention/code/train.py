
# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# # Options 

# In[1]:


base_path = 'path to your base directory'
opt = {
    'num_layers' : 2,
    'droprnn' : 0.1,
    'dropout' : 0.3,
    'dropattn' : 0,
    'seq_per_img' : 1,
    'cnn_name' : 'resnet50',
    'save_model' : base_path+'models/trained',
    'hidden_size' : 1024,
    'wdim' : 256,
    'num_img_attn' : 4,
    'num_dense_attn' : 4,
    'num_predict_attn' : 4,
    'num_none' : 3,
    'num_seq' : 3,
    'predict_type' : 'cat_attn',
    'gpus' : [0],
    'log_interval' : 100,
    'record_step' : 1,
    'num_epoch' : 16,
    'patience' : 3,
    'save_freq' : 8,
    'trainval' : 1,
    'seed' : 12345,
    'data_path' : base_path + 'data/',
    'data_name' : 'cocotrainval',
    'img_name' : 'cocoimages',
    'num_workers' : 8,
    'batch_size' : 48,
    'word_vectors' : base_path + 'data/glove_840B.pt',
    'lr' : 0.001,
    'gamma' : 0.5,
    'step_size' : 7,
    'weight_decay' : 0.0001,
    'size_scale' : (448, 448),
    'train_from' : base_path + '/models/v1_trainval_trained.pt',
    'max_grad_norm' : None,
    'use_h5py' : False,
    'shuffle' : True,
    'pin_memory' : False,
    'drop_last' : False,
    'use_thread' : True,
    'use_rcnn' : False,
}

# # Helper Functions 

# In[2]:

import torch
import torch.nn as nn
import argparse
import json
import sys
import torch.optim.lr_scheduler as lr_scheduler

import pdb

from tensorboardX import SummaryWriter
from dense_coattn.model import DCN
from dense_coattn.modules import LargeEmbedding
from dense_coattn.data import Dataset, DataLoader
from dense_coattn.util import Initializer, Meter, Timer, Saver
from dense_coattn.optim import OptimWrapper, Adam, SGD
from dense_coattn.evaluate import Accuracy
from dense_coattn.cost import BinaryLoss, LossCompute

def move_to_cuda(tensors, devices=None):
    if devices is not None:
        if len(devices) >= 1:
            cuda_tensors = []
            for tensor in tensors:
                if tensor is not None:
                    cuda_tensors.append(tensor.cuda(devices[0], async=True))
                else:
                    cuda_tensors.append(None)
            return tuple(cuda_tensors)
    return tensors

def trainEpoch(epoch, dataloader, model, criterion, evaluation, optim, opt, writer):
    model.train()
    loss_record = [Meter() for _ in range(3)]
    accuracy_record = [Meter() for _ in range(3)]
    timer = Timer()

    timer.tic()
    optim.step_epoch()
    for i, batch in enumerate(dataloader):
        if not opt['use_rcnn']:
            img, ques, ques_mask, _, ans_idx = batch
        else:
            img, ques, img_mask, ques_mask, _, ans_idx = batch

        img = torch.tensor(img, requires_grad=False)
        img_mask = None
        ques = torch.tensor(ques, requires_grad=False)
        ques_mask = torch.tensor(ques_mask)
        ans_idx = torch.tensor(ans_idx)

        img, img_mask, ques, ques_mask, ans_idx = move_to_cuda((img, img_mask, ques, ques_mask, ans_idx), devices=opt['gpus'])
        ques = model.word_embedded(ques)
        ques = torch.tensor(ques.data)

        optim.zero_grad()
        score = model(img, ques, img_mask, ques_mask, is_train=True)

        loss = criterion(score, ans_idx)
        loss.backward()
        accuracy = evaluation(torch.tensor(score.data, requires_grad=False), torch.tensor(ans_idx.data, requires_grad=False))
        _, ratio, updates, params = optim.step()

        for j in range(3):
            loss_record[j].update((loss.item() / opt['batch_size']))
            accuracy_record[j].update(accuracy.item())

        if ratio is not None:
            writer.add_scalar("statistics/update_to_param_ratio", ratio, global_step=(epoch*len(dataloader) + i))
            writer.add_scalar("statistics/absolute_updates", updates, global_step=(epoch*len(dataloader) + i))
            writer.add_scalar("statistics/absolute_params", params, global_step=(epoch*len(dataloader) + i))
        
        if (i + 1) % 10 == 0:
            writer.add_scalar("iter/train_loss", loss_record[0].avg, global_step=(epoch*len(dataloader) + i))
            writer.add_scalar("iter/train_accuracy", accuracy_record[0].avg, global_step=(epoch*len(dataloader) + i))
            loss_record[0].reset()
            accuracy_record[0].reset()

        if (i + 1) % opt['log_interval'] == 0:
            print("Epoch %5d; iter %6i; loss: %8.2f; accuracy: %8.2f; %6.0fs elapsed" %
                  (epoch, i+1, loss_record[1].avg, accuracy_record[1].avg, timer.toc(average=False)))
            loss_record[1].reset()
            accuracy_record[1].reset()
            timer.tic()

    writer.add_scalar("epoch/train_loss", loss_record[2].avg, global_step=epoch)
    writer.add_scalar("epoch/train_accuracy", accuracy_record[2].avg, global_step=epoch)

    return loss_record[2].avg, accuracy_record[2].avg


def evalEpoch(epoch, dataloader, model, criterion, evaluation, opt, writer):
    model.eval()
    total_loss, total_accuracy = Meter(), Meter()

    for i,batch in enumerate(dataloader):
        img, ques, ques_mask, _, ans_idx = batch
        
        img = torch.tensor(img, requires_grad=False)
        img_mask = None
        ques = torch.tensor(ques, requires_grad=False)
        ques_mask = torch.tensor(ques_mask, requires_grad=False)
        ans_idx = torch.tensor(ans_idx, requires_grad=False)

        img, img_mask, ques, ques_mask, ans_idx = move_to_cuda((img, img_mask, ques, ques_mask, ans_idx), devices=opt['gpus'])
        ques = model.word_embedded(ques)

        score = model(img, ques, img_mask, ques_mask, is_train=False)
        accuracy = evaluation(score, ans_idx)
        loss = criterion(score, ans_idx)

        total_loss.update((loss.item() / opt['batch_size']))
        total_accuracy.update(accuracy.item())

        if (i + 1) % opt['log_interval'] == 0:
            print("VALIDATION: Epoch %5d; iter %6i" %(epoch, i+1))
    writer.add_scalar("epoch/val_loss", total_loss.avg, global_step=epoch)
    writer.add_scalar("epoch/val_accuracy", total_accuracy.avg, global_step=epoch)

    return total_loss.avg, total_accuracy.avg


def trainModel(trainLoader, valLoader, model, criterion, evaluation, optim, opt):
    best_accuracy = None
    bad_counter = None
    history = None

    if valLoader is not None:
        best_accuracy = 0
        bad_counter = 0
        history = []
    writer = SummaryWriter(log_dir="logs/%s" % opt['save_model'].split("/")[-1])
    print('No of train batches: ', len(trainLoader))
    if valLoader is not None: print('No of test batches: ', len(valLoader))
    for epoch in range(opt['num_epoch']):
        print("----------------------------------------------EPOCH: ",epoch)
    
        train_loss, train_accuracy = trainEpoch(epoch, trainLoader, model, criterion, 
        evaluation, optim, opt, writer)
        print("Train loss: %10.4f, accuracy: %5.2f" % (train_loss, train_accuracy))

        is_parallel = True if len(opt['gpus']) > 1 else False
        model_state_dict = Saver.save_state_dict(model, excludes=["word_embedded"], is_parallel=is_parallel)
        Saver.save_model(model_state_dict, opt, epoch, best_accuracy, history, save_type=0)

        if valLoader is not None:
            val_loss, val_accuracy = evalEpoch(epoch, valLoader, model, criterion, evaluation, opt, writer)
            print("Val loss: %10.4f, accuracy: %5.2f" % (val_loss, val_accuracy))
            history.append(val_accuracy)

            if best_accuracy <= val_accuracy:
                best_accuracy = val_accuracy
                Saver.save_model(model_state_dict, opt, epoch, best_accuracy, history, save_type=1)
                bad_counter = 0

            if (len(history) > opt['patience']) and val_accuracy <= torch.torch.tensor(history[:-opt['patience']]).max():
                bad_counter += 1
                if bad_counter > opt['patience']:
                    print("Early Stop!")
                    break
        if ((epoch + 1) % opt['save_freq'] == 0):
            Saver.save_model(model_state_dict, opt, epoch, best_accuracy, history, save_type=2)
    writer.close()


# # Training Code 

# In[ ]:


Initializer.manual_seed(opt['seed'])
print("Constructing the dataset...")
if opt['trainval'] == 0:
    trainset = Dataset(opt['data_path'], opt['data_name'], "train", opt['seq_per_img'], opt['img_name'], 
        opt['size_scale'], use_h5py=opt['use_h5py'])
    trainLoader = DataLoader(trainset, batch_size=opt['batch_size'], shuffle=opt['shuffle'], 
        num_workers=opt['num_workers'], pin_memory=opt['pin_memory'], drop_last=opt['drop_last'], use_thread=opt['use_thread'])

    valset = Dataset(opt['data_path'], opt['data_name'], "val", opt['seq_per_img'], opt['img_name'], 
        opt['size_scale'], use_h5py=opt['use_h5py'])
    valLoader = DataLoader(valset, batch_size=opt['batch_size'], shuffle=opt['shuffle'],
        num_workers=opt['num_workers'], pin_memory=opt['pin_memory'], drop_last=opt['drop_last'], use_thread=opt['use_thread'])
else:
    trainset = Dataset(opt['data_path'], opt['data_name'], "trainval", opt['seq_per_img'], opt['img_name'], 
        opt['size_scale'], use_h5py=opt['use_h5py'])
    trainLoader = DataLoader(trainset, batch_size=opt['batch_size'], shuffle=opt['shuffle'], 
        num_workers=opt['num_workers'], pin_memory=opt['pin_memory'], drop_last=opt['drop_last'], use_thread=opt['use_thread'])

    valset = None
    valLoader = None

idx2word = trainset.idx2word
ans_pool = trainset.ans_pool
ans_pool = torch.from_numpy(ans_pool)

print("Building model...")
word_embedded = LargeEmbedding(len(idx2word), 300, padding_idx=0, devices=opt['gpus'])
word_embedded.load_pretrained_vectors(opt['word_vectors'])

num_ans = ans_pool.size(0)
model = DCN(opt, num_ans)

criterion = BinaryLoss()
evaluation = Accuracy()

dict_checkpoint = opt['train_from']
if dict_checkpoint:
    print("Loading model from checkpoint at %s" % dict_checkpoint)
    checkpoint = torch.load(dict_checkpoint)
    model.load_state_dict(checkpoint["model"])

if len(opt['gpus']) >= 1:
    model.cuda(opt['gpus'][0])

if len(opt['gpus']) > 1:
    model = nn.DataParallel(model, opt['gpus'], dim=0)
model.word_embedded = word_embedded

print('Getting optimizer')
optimizer = Adam(list(filter(lambda x: x.requires_grad, model.parameters())), lr=opt['lr'],
    weight_decay=opt['weight_decay'], record_step=opt['record_step'])
scheduler = lr_scheduler.StepLR(optimizer, opt['step_size'], gamma=opt['gamma'])
optim_wrapper = OptimWrapper(optimizer, scheduler)

nparams = []
named_parameters = model.module.named_parameters() if len(opt['gpus']) > 1 else model.named_parameters()
for name, param in named_parameters:
    if not (name.startswith("resnet") or name.startswith("word_embedded") or name.startswith("ans")):
        nparams.append(param.numel())
print("* Number of parameters: %d" % sum(nparams))

checkpoint = None
timer = Timer()
timer.tic()
try:
    with torch.cuda.device(opt['gpus'][0]):
        print('Training model....')
        trainModel(trainLoader, valLoader, model, criterion, evaluation, optim_wrapper, opt)
except KeyboardInterrupt:
    print("It toke %.2f hours to train the network" % (timer.toc() / 3600))
    sys.exit("Training interrupted")

print("It toke %.2f hours to train the network" % (timer.toc() / 3600))