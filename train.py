import os
import pdb
import numpy as np
from tqdm import tqdm
from jiwer import wer
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
from option import get_parser


def train(opt):
    utils.init_log_dir(opt)
    writer = SummaryWriter('./save/{}/tb'.format(opt.name))

    train_set, val_set, _ = utils.get_dataset(opt)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.num_workers, drop_last=True)

    val_loader = DataLoader(val_set, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.num_workers)

    encode, middle, decode = utils.get_model(opt)
    if opt.gpu:
        encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()
    if opt.gpus:
        encode, middle, decode = nn.DataParallel(
            encode), nn.DataParallel(middle), nn.DataParallel(decode)

    optimizer = optim.Adam([{'params': encode.parameters()},
                            {'params': middle.parameters()},
                            {'params': decode.parameters()}], opt.lr, weight_decay=opt.weight_decay)

    scheduler = utils.AdjustLR(
        optimizer, [opt.lr, opt.lr, opt.lr], sleep_epochs=5, half=5)

    best_val_loss = 1e10
    for epoch in range(opt.epoches):
        encode.train()
        middle.train()
        decode.train()
        scheduler.step(epoch)
        for step, pack in enumerate(train_loader):
            v = pack[0]
            align = pack[1]
            if opt.gpu:
                v = v.cuda()
                align = align.cuda()

            embeddings = encode(v)
            embeddings, enc_mask = middle(embeddings)

            loss, loss_real, _ = decode(embeddings, align[:, :-1], align[:, 1:], enc_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % 10 == 0:
                print('epoch:{} step:{}/{} train_loss:{:.4f} train_loss(real):{:.4f}'.format(
                    epoch, (step+1), len(train_loader), loss.item(), loss_real.item()))
                writer.add_scalar('train-loss', loss.item(), epoch * len(train_loader) + step + 1)
                writer.add_scalar('train-loss-real', loss_real.item(), epoch * len(train_loader) + step + 1)

        val_loss = 0.
        encode.eval()
        middle.eval()
        decode.eval()
        with torch.no_grad():
            for step, pack in enumerate(val_loader):
                v = pack[0]
                align = pack[1]
                if opt.gpu:
                    v = v.cuda()
                    align = align.cuda()

                embeddings = encode(v)
                embeddings, enc_mask = middle(embeddings)

                loss, loss_real, _ = decode(embeddings, align[:, :-1], align[:, 1:], enc_mask)
                val_loss += loss_real.item()

            val_loss /= len(val_loader)
            print('>>>>>>>>>>val loss:{:.4f}>>>>>>>>>>'.format(val_loss), end=' ')
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                print('best')
                torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
                            'mid_net': middle.module.state_dict() if opt.gpus else middle.state_dict(),
                            'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
                            'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'best.pth.tar'.format(epoch, step+1)))
            else:
                print('')


def test(opt):
    _, _, test_set = utils.get_dataset(opt)

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

    wer_list = []
    person_list = []
    with torch.no_grad():
        for step, pack in enumerate(tqdm(test_loader)):
            v = pack[0]
            align = pack[1]
            text = pack[3]
            items = pack[4]
            if opt.gpu:
                v = v.cuda()

            embeddings = encode(v)
            embeddings, enc_mask = middle(embeddings)

            res = torch.zeros(embeddings.size(0), embeddings.size(1)).cuda()
            input_align = torch.zeros(embeddings.size(0), embeddings.size(1)).cuda()
            input_align[:, 0] = test_set.character_dict['*']
            for i in range(embeddings.size(1)):
                digits = decode.forward_infer(embeddings, input_align, enc_mask)
                pred = torch.argmax(digits, -1)
                res[:, i] = pred[:, i]
                if i < embeddings.size(1)-1:
                    input_align[:, i+1] = pred[:, i]

            pred = list(map(lambda x: ''.join([test_set.idx_dict[i.item()] for i in x]).replace('^', ''), res))
            wer_list.extend([wer(p, t) for p, t in zip(pred, text)])
            person_list += [x.split('/')[-2] for x in items]

    wer_list = np.array(wer_list)
    person_list = np.array(person_list)
    print('overall wer:{:.4f}'.format(np.mean(wer_list)))
    for person in list(set(person_list)):
        print('{} wer:{:.4f}'.format(person, np.mean(wer_list[person_list == person])))


if __name__ == '__main__':
    opt = get_parser()
    train(opt)
    test(opt)