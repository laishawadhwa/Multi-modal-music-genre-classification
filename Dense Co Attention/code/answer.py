
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import json

from dense_coattn.data import Dataset, DataLoader
from dense_coattn.modules import LargeEmbedding
from dense_coattn.model import DCN, DCNWithAns

base_path = 'path to the directory with model '
opt = {
    "num_layers" : 2,
    "seq_per_img" : 1,
    "droprnn" : 0.1,
    "dropout" : 0.3,
    "dropattn" : 0,
    "cnn_name" : "resnet50",
    "hidden_size" : 1024,
    "wdim" : 256,
    "num_img_attn" : 4,
    "num_dense_attn" : 4,
    "num_predict_attn" : 4,
    "num_none" : 3,
    "num_seq" : 3,
    "predict_type" : "cat_attn",
    "gpus" : [0],
    "data_path" : base_path + 'data/',
    "data_name" : "cocotrainval",
    "img_name" : "cocoimages",
    "num_workers" : 4,
    "batch" : 48,
    "word_vectors" : base_path + 'data/glove_840B.pt',
    "train_from" : base_path + '/models/trained_15.pt',
    "save_file" : "results_vqa",
    "use_h5py" : False,
    "use_rcnn" : False,
    "use_thread" : True,
    "size_scale" : (448, 448),
}

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


def answer(dataloader, model, idx2ans, opt):
    model.eval()
    num_batches = len(dataloader)
    answers = []
    scores = []
    for i, batch in enumerate(dataloader):
        img, ques, ques_mask, ques_idx = batch
            
        img = torch.tensor(img, requires_grad=False)
        img_mask = None
        ques = torch.tensor(ques, requires_grad=False)
        ques_mask = torch.tensor(ques_mask, requires_grad=False)

        img, img_mask, ques, ques_mask = move_to_cuda((img, img_mask, ques, ques_mask), devices=opt['gpus'])
        ques = model.word_embedded(ques)

        score = model(img, ques, img_mask, ques_mask, is_train=False)
        _, inds = torch.sort(score, dim=1, descending=True)

        for j in range(ques_idx.size(0)):
            answers.append({"question_id": ques_idx[j], "answer": idx2ans[inds.data[j, 0]]})
        if i % 10 == 0:
            print("processing %i / %i" % (i, num_batches))

    with open("%s.json" % (opt['save_file']), "w") as file:
        json.dump(answers, file)
        
    print("Done!")


print("Constructing the dataset...")
testset = Dataset(opt['data_path'], opt['data_name'], "test", opt['seq_per_img'], opt['img_name'],
    opt['size_scale'], use_h5py=opt['use_h5py'])
testLoader = DataLoader(testset, batch_size=opt['batch'], shuffle=False, 
    num_workers=opt['num_workers'], pin_memory=True, drop_last=False, use_thread=opt['use_thread'])

idx2word = testset.idx2word
idx2ans = testset.idx2ans
ans_pool = testset.ans_pool
ans_pool = torch.from_numpy(ans_pool)

print("Building model...")
word_embedded = LargeEmbedding(len(idx2word), 300, padding_idx=0, devices=opt['gpus'])
word_embedded.load_pretrained_vectors(opt['word_vectors'])

num_ans = ans_pool.size(0)
model = DCN(opt, num_ans)

dict_checkpoint = opt['train_from']
if dict_checkpoint:
    print("Loading model from checkpoint at %s" % dict_checkpoint)
    checkpoint = torch.load(dict_checkpoint)
    model.load_state_dict(checkpoint["model"])

if len(opt['gpus']) >= 1:
    model.cuda(opt['gpus'][0])
model.word_embedded = word_embedded

print("Generating answers...")
with torch.cuda.device(opt['gpus'][0]):
    answer(testLoader, model, idx2ans, opt)
