import os
import math
import cv2
import shutil
import requests
import numpy as np
import torch


def oned_smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size."
               )

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (
            ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def get_imgs_from_video(video, ext='jpg', RGB=False):
    frames = []
    if os.path.isdir(video):
        frames = sorted(glob.glob(os.path.join(video, '*.{}'.format(ext))),
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        frames = [cv2.imread(f) for f in frames]
    else:
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

    frames = np.array(frames)
    if RGB:
        return frames[..., ::-1]
    else:
        return frames


def get_dataset(options):
    if options.dataset == 'grid':
        from data.grid import GridSeq2Seq
        dataset_train = GridSeq2Seq(options, mode='train')
        dataset_val = GridSeq2Seq(options, mode='val')
        dataset_test = GridSeq2Seq(options, mode='test')
    elif options.dataset == 'grid_raw':
        from data.grid import GridRaw
        dataset_train = GridRaw(options, mode='train')
        dataset_val = GridRaw(options, mode='val')
        dataset_test = GridRaw(options, mode='test')
    elif options.dataset == 'lrw':
        from data.lrw import LRWClassification
        dataset_train = LRWClassification(options, mode='train')
        dataset_val = LRWClassification(options, mode='val')
        dataset_test = LRWClassification(options, mode='test')
    else:
        raise

    return (dataset_train, dataset_val, dataset_test)


def init_log_dir(opt):
    if os.path.exists(os.path.join('./save', opt.name)):
        print('dir exist, delete?')
        x = input()
        if x == 'y':
            shutil.rmtree(os.path.join('./save', opt.name))
        else:
            raise

    os.mkdir(os.path.join('./save', opt.name))
    with open(os.path.join('./save', opt.name, 'option.txt'), "a") as f:
        for k, v in vars(opt).items():
            f.write('{} -> {}\n'.format(k, v))

    os.mkdir(os.path.join('./save', opt.name, 'check'))
    os.mkdir(os.path.join('./save', opt.name, 'img'))
    os.mkdir(os.path.join('./save', opt.name, 'tb'))


def get_model(options):
    if options.encode == '233':
        from model.encode import D3Conv_Res34_233
        encode = D3Conv_Res34_233(options)
    elif options.encode == 'pass':
        from model.middle import Pass
        encode = Pass(options)
    else:
        encode = None

    if options.middle == 'tc':
        from model.middle import MidTemporalConv
        mid_net = MidTemporalConv(options)
    elif options.middle == 'atten':
        from model.middle import EncodeAttention
        mid_net = EncodeAttention(options)
    else:
        mid_net = None

    if options.decode == 'pass':
        from model.middle import Pass
        decode = Pass(options)
    elif options.decode == 'atten':
        from model.decode import DecodeAtten
        decode = DecodeAtten(options)
    else:
        decode = None

    return encode, mid_net, decode


class AdjustLR(object):
    def __init__(self, optimizer, init_lr, sleep_epochs=3, half=5):
        super(AdjustLR, self).__init__()
        self.optimizer = optimizer
        self.sleep_epochs = sleep_epochs
        self.half = half
        self.init_lr = init_lr

    def step(self, epoch):
        if epoch >= self.sleep_epochs:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                new_lr = self.init_lr[idx] * \
                    math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
                param_group['lr'] = new_lr


def one_hot(opt, indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if opt.gpu:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def crop_lmark_center_mouth(img, lmark):
    def center_ab(a, b, c, d):
        return [(b[0]+a[0])//2, (b[1]+a[1])//2]

    lmark_2 = lmark[:, :2]
    c = center_ab(lmark_2[48], lmark_2[54], lmark_2[51], lmark_2[57])
    return c
