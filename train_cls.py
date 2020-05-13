import os
import pdb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
from option import get_parser


def train(opt):
    utils.init_log_dir(opt)

    train_set, _, _ = utils.get_dataset(opt)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.num_workers, drop_last=True)

    encode, middle, decode = utils.get_model(opt)
    if opt.gpu:
        encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()
    if opt.gpus:
        encode, middle, decode = nn.DataParallel(
            encode), nn.DataParallel(middle), nn.DataParallel(decode)

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam([{'params': encode.parameters()},
                            {'params': middle.parameters()},
                            {'params': decode.parameters()}], opt.lr, weight_decay=opt.weight_decay)

    scheduler = utils.AdjustLR(
        optimizer, [opt.lr, opt.lr, opt.lr], sleep_epochs=5, half=5)

    for epoch in range(opt.epoches):
        scheduler.step(epoch)
        for step, pack in enumerate(train_loader):
            v = pack[0]
            align = pack[1]
            if opt.gpu:
                v = v.cuda()
                align = align.cuda()

            embeddings = encode(v)
            embeddings = middle(embeddings)

            digits = decode(embeddings)

            if opt.loss_smooth == 0:
                loss = loss_func(digits, align)
            else:
                smoothed_one_hot = utils.one_hot(opt, align, opt.out_channel)
                smoothed_one_hot = smoothed_one_hot * \
                    (1 - opt.loss_smooth) + (1 - smoothed_one_hot) * \
                    opt.loss_smooth / (opt.out_channel - 1)

                log_prb = F.log_softmax(digits, dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % 10 == 0:
                res = torch.argmax(digits.detach(), -1).cpu().numpy()
                top_1_cls = np.mean(res == align.cpu().numpy())
                print('epoch:{} step:{}/{} train_loss:{:.4f} train_acc:{:.4f}'.format(
                    epoch, (step+1), len(train_loader), loss.item(), top_1_cls))

        torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
                    'mid_net': middle.module.state_dict() if opt.gpus else middle.state_dict(),
                    'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
                    'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'model_{}_{}.pth.tar'.format(epoch, step+1)))


def test(opt):
    _, val_set, test_set = utils.get_dataset(opt)

    val_loader = DataLoader(val_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size,
                             shuffle=False, num_workers=opt.num_workers, drop_last=False)

    encode, middle, decode = utils.get_model(opt)
    checkpoint = torch.load(
        './save/lrw/check/model_12_4200.pth.tar', map_location='cpu')
    encode.load_state_dict(checkpoint['encode'])
    middle.load_state_dict(checkpoint['mid_net'])
    decode.load_state_dict(checkpoint['decode'])
    encode, middle, decode = encode.eval(), middle.eval(), decode.eval()
    if opt.gpu:
        encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

    assert not opt.gpus

    top1 = []
    top5 = []
    with torch.no_grad():
        for step, pack in enumerate(tqdm(val_loader)):
            v = pack[0]
            align = pack[1]
            if opt.gpu:
                v = v.cuda()
                align = align.cuda()

            v = v.transpose(1, 2)
            v = v.reshape([-1]+list(v.shape[2:]))
            embeddings = encode(v)
            embeddings = middle(embeddings)

            digits = decode(embeddings)
            digits = digits.reshape([-1, 10, digits.size(-1)]).mean(1).cpu()

            top1.append(align in torch.topk(digits, 1)[1])
            top5.append(align in torch.topk(digits, 5)[1])

    print('val top1:{}'.format(np.mean(top1)))
    print('val top5:{}'.format(np.mean(top5)))

    top1 = []
    top5 = []
    with torch.no_grad():
        for step, pack in enumerate(tqdm(test_loader)):
            v = pack[0]
            align = pack[1]
            if opt.gpu:
                v = v.cuda()
                align = align.cuda()

            v = v.transpose(1, 2)
            v = v.reshape([-1]+list(v.shape[2:]))
            embeddings = encode(v)
            embeddings = middle(embeddings)

            digits = decode(embeddings)
            digits = digits.reshape([-1, 10, digits.size(-1)]).mean(1).cpu()

            top1.append(align in torch.topk(digits, 1)[1])
            top5.append(align in torch.topk(digits, 5)[1])

    print('test top1:{}'.format(np.mean(top1)))
    print('test top5:{}'.format(np.mean(top5)))


def encode_grid(opt):
    train_set, _, test_set = utils.get_dataset(opt)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size,
                             shuffle=False, num_workers=opt.num_workers, drop_last=False)

    encode, middle, decode = utils.get_model(opt)
    checkpoint = torch.load(opt.load, map_location='cpu')
    encode.load_state_dict(checkpoint['encode'])
    middle.load_state_dict(checkpoint['mid_net'])
    decode.load_state_dict(checkpoint['decode'])
    encode, middle, decode = encode.eval(), middle.eval(), decode.eval()
    if opt.gpu:
        encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

    assert not opt.gpus

    with torch.no_grad():
        for step, pack in enumerate(tqdm(train_loader)):
            v = pack[0]
            path = pack[1]
            if opt.gpu:
                v = v.cuda()

            v = v.transpose(1, 2)
            v = v.reshape([-1]+list(v.shape[2:]))
            embeddings = encode(v)
            embeddings = embeddings.reshape([-1, 10]+list(embeddings.shape[1:])).mean(1).cpu().numpy()
            for emb, p in zip(embeddings, path):
                np.savez(p.replace('_mouth.npz', '_233feat.npz'), data=emb)

    with torch.no_grad():
        for step, pack in enumerate(tqdm(test_loader)):
            v = pack[0]
            path = pack[1]
            if opt.gpu:
                v = v.cuda()

            v = v.transpose(1, 2)
            v = v.reshape([-1]+list(v.shape[2:]))
            embeddings = encode(v)
            embeddings = embeddings.reshape([-1, 10]+list(embeddings.shape[1:])).mean(1).cpu().numpy()
            for emb, p in zip(embeddings, path):
                np.savez(p.replace('_mouth.npz', '_233feat.npz'), data=emb)


if __name__ == '__main__':
    opt = get_parser()
    train(opt)
    test(opt)
    encode_grid(opt)
