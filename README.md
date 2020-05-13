# LipRead-seq2seq

This is an unofficial (PyTorch) implementation for the paper [Deep Lip Reading: A comparison of models and an online application](http://www.robots.ox.ac.uk/~vgg/publications/2018/Afouras18b/afouras18b.pdf). **Please note since the MV-LRS dataset is non-public, we only do the experiments on GRID dataset with the same model structure and training process.**

## Data

GRID dataset:

1. Download [GRID dataset](http://spandh.dcs.shef.ac.uk/gridcorpus/#downloads) (high video version). Move it to ./dataset/grid/, or you can move it to somewhere else and modify  --grid_root in option.py.
2. We first crop the entire face from each video (with corresponding landmarks). Uncomment *crop_video(opt)* in grid.py and comment others.

```
python data/grid.py
```

3. Then we crop the mouth region of each sample. Uncomment *crop_mouth(opt)* in grid.py and comment others.

```
python data/grid.py
```

LRW dataset: we follow the preprocessing prodecure in [end-to-end-lipreading](https://github.com/mpc001/end-to-end-lipreading). And also the part of code is from it.

1. Download&unzip [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) and move it to ./dataset/lrw/, or you can move it to somewhere else and modify  --lrw_root in option.py.
2. Crop the mouth area of each video.

```shell
python data/lrw.py
```

## Encode

We first train an encode network with LRW dataset. The network will be used later to encode features of GRID videos. **If you want to do experiments on LRS2/LRS3 dataset, you need to have MV-LRS(w) dataset described in the paper and process it in the same way as LRW. Then you can train these two dataset jointly to get a good encode network for profiles.**

```shell
python train_cls.py --name lrw --encode 233 --middle tc --decode pass --num_workers 16 --gpu --batch_size 32
```

If you have multiple GPUs, you can try

```shell
python train_cls.py --name lrw --encode 233 --middle tc --decode pass --num_workers 16 --gpu --gpus --batch_size 128
```

You can test the model on val and test dataset. You need to comment *train(opt)* and uncomment *test(opt)* in *train_cls.py*. We use ten-crop to augment the test process.

```shell
python train_cls.py --name lrw --encode 233 --middle tc --decode pass --num_workers 16 --gpu --batch_size 1 --load ./save/lrw/check/model_12_4200.pth.tar
```

The test result of our trained model is shown bellow.

|                | TOP1(%) | TOP5(%) |
| :------------: | :-----: | :-----: |
| validation set |  73.15  |  92.44  |
|    test set    |  72.25  |  92.21  |

## Encode GRID

After training the classification model with LRW dataset. We use the encoding network to encode the preprocessed GRID dataset. Comment *train(opt)* and *test(opt)*. Uncomment *encode_grid(opt)*.

```shell
CUDA_VISIBLE_DEVICES=0 python train_cls.py --dataset grid_raw --gpu --batch_size 8 --num_workers 16 --load ./save/lrw/check/model_12_4200.pth.tar
```

## Seq2Seq

Train Grid dataset in seq2seq mode. Uncomment *train(opt)* while comment *test(opt)*.

```shell
CUDA_VISIBLE_DEVICES=1 python train.py --name grid --dataset grid --num_workers 16 --batch_size 96 --encode pass --middle atten --decode atten --out_channel 28 --gpu --validation_batch 2
```

Test the model. Uncomment *test(opt)* and comment *train(opt)*.

```shell
CUDA_VISIBLE_DEVICES=1 python train.py --dataset grid --num_workers 16 --batch_size 16 --encode pass --middle atten --decode atten --out_channel 28 --gpu --validation_batch 2 --load ./save/grid/check/best.pth.tar
```

The test result is shown below.

|        | overall |  s1   |  s2   |  s20  |  s22  |
| :----: | :-----: | :---: | :---: | :---: | :---: |
| WER(%) |  42.49  | 58.29 | 32.93 | 43.45 | 35.42 |

## Improvement

We only implement the basic framework in the paper. To improve the performance, you can do the following things.

- Train a better encode network with LRW dataset. The result in [end-to-end-lipreading](https://github.com/mpc001/end-to-end-lipreading) is 10% higher than ours, which I think is very important.
- Add data augmentation method mentioned in the paper.
- Add language model mentioned in the paper.
- Add beam search.
- Adjust hyperparameters carefully when training.