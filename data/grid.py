import os
import cv2
import pdb
import sys
import glob
import random
import numpy as np
from tqdm import tqdm
import face_alignment
from PIL import Image

import torch
import torchvision.transforms as transforms

sys.path.append('./')
import utils
from option import get_parser

def crop_video(opt):
	def crop_img(frame, lStart=36, lEnd=41, rStart=42, rEnd=47):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		lmark = fa.get_landmarks(gray)[0]

		leftEyePts = lmark[lStart:lEnd]
		rightEyePts = lmark[rStart:rEnd]

		leftEyeCenter = leftEyePts.mean(axis=0)
		rightEyeCenter = rightEyePts.mean(axis=0)
		max_v = np.amax(lmark, axis=0)
		min_v = np.amin(lmark, axis=0)

		max_x, max_y = max_v[0], max_v[1]
		min_x, min_y = min_v[0], min_v[1]
		dis = max(max_y - min_y, max_x - min_x)

		two_eye_center = (leftEyeCenter + rightEyeCenter)/2
		center_y, center_x = two_eye_center[0], two_eye_center[1]

		return center_y, center_x, dis, lmark

	def crop(v):
		rois = []
		lmarks = []
		x_list, y_list, dis_list, frames = [], [], [], []
		cap = cv2.VideoCapture(v)
		img_size = None
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == True:
				center_y, center_x, dis, lmark = crop_img(frame)
				y_list.append(center_y)
				x_list.append(center_x)
				dis_list.append(dis)
				frames.append(frame)
				lmarks.append(lmark)
			else:
				break

		x_list, y_list, dis_list = np.array(
			x_list), np.array(y_list), np.array(dis_list)
		dis = np.mean(dis_list)
		side_length = int((205 * dis / 90))
		top_left_x = x_list - (80 * dis / 90)
		top_left_y = y_list - (100 * dis / 90)
		top_left_x = utils.oned_smooth(top_left_x)
		top_left_y = utils.oned_smooth(top_left_y)

		for i in range(len(x_list)):
			if top_left_x[i] < 0 or top_left_y[i] < 0:
				img_size = frames[i].shape
				tempolate = np.zeros(
					(img_size[0] * 2, img_size[1] * 2, 3), np.uint8)
				tempolate_middle = [
					int(tempolate.shape[0]/2), int(tempolate.shape[1]/2)]
				middle = [int(img_size[0]/2), int(img_size[1]/2)]
				tempolate[tempolate_middle[0] - middle[0]:tempolate_middle[0]+middle[0],
						  tempolate_middle[1]-middle[1]:tempolate_middle[1]+middle[1], :] = frames[i]
				top_left_x[i] = top_left_x[i] + tempolate_middle[0] - middle[0]
				top_left_y[i] = top_left_y[i] + tempolate_middle[1] - middle[1]
				roi = tempolate[int(top_left_x[i]):int(
					top_left_x[i]) + side_length, int(top_left_y[i]):int(top_left_y[i]) + side_length]
			else:
				roi = frames[i][int(top_left_x[i]):int(
					top_left_x[i]) + side_length, int(top_left_y[i]):int(top_left_y[i]) + side_length]

			roi = cv2.resize(roi, (256, 256))
			rois.append(roi)

		return rois, lmarks

	fa = face_alignment.FaceAlignment(
		face_alignment.LandmarksType._2D, flip_input=False)  # ,  device='cpu' )
	videos = glob.glob(os.path.join(
		opt.grid_root, 's*', 'video', 'mpg_6000', '*.mpg'))
	for v in tqdm(videos):
		crop_v, lmarks = crop(v)
		np.savez(v[:-4]+'_crop.npz', data=crop_v)
		np.save(v[:-4]+'_lmark.npy', data=lmarks)


def crop_mouth(opt):
	videos = glob.glob(os.path.join(
		opt.grid_root, 's*', '*_original.npy'))

	for v in tqdm(videos):
		frames = utils.get_imgs_from_video(
			v.replace('_original.npy', '_crop.mp4'))
		lmarks = np.load(v)

		try:
			assert len(frames) == len(lmarks)

			mouth_center_list = np.asarray([utils.crop_lmark_center_mouth(
				img, lmarks[i]) for i, img in enumerate(frames)])
			crop_center_list_smooth_x = utils.oned_smooth(mouth_center_list[:, 0])[
				:len(mouth_center_list[:, 0])]
			crop_center_list_smooth_y = utils.oned_smooth(mouth_center_list[:, 1])[
				:len(mouth_center_list[:, 1])]
			mouth_center_list = list(
				zip(crop_center_list_smooth_x, crop_center_list_smooth_y))

			crop_img_list = [img[int(mouth_center_list[i][1])-40:int(mouth_center_list[i][1])+40, int(
				mouth_center_list[i][0])-40:int(mouth_center_list[i][0])+40] for i, img in enumerate(frames)]
			for img in crop_img_list:
				assert img.shape[0] == img.shape[1] == 80
		except:
			print(v)

		np.savez(v[:-len('_original.npy')]+'_mouth.npz', data=crop_img_list)


class GridRaw:
	def __init__(self, args, mode='train'):
		self.mode = mode
		self.args = args
		self.data = glob.glob(os.path.join(
			args.grid_root, 's*', '*_mouth.npz'))

		if mode == 'train':
			self.cur_data = [x for x in self.data if x.split(
				'/')[-2] not in ['s1', 's2', 's20', 's22']]
		elif mode == 'val':
			self.cur_data = None
		else:
			self.cur_data = [x for x in self.data if x.split(
				'/')[-2] in ['s1', 's2', 's20', 's22']]

		self.transform = self.get_transform()

	def __getitem__(self, index):
		item = self.cur_data[index]

		video = np.load(item)['data']
		if len(video) < 75:
			padf = 75 - len(video)
			video = np.concatenate(
				[video, np.zeros([padf]+list(video.shape[1:])).astype(np.uint8)])
		elif len(video) > 75:
			video = video[:75]

		array = [Image.fromarray(f) for f in video]
		array = [self.transform(im) for im in array]
		array = torch.stack(array)

		return array, item

	def __len__(self):
		return len(self.cur_data)

	def get_transform(self):
		return transforms.Compose([
			transforms.Grayscale(),
			transforms.Resize((96, 96)),
			transforms.TenCrop((88, 88)),
			transforms.Lambda(lambda crops: torch.stack(
				[transforms.ToTensor()(crop) for crop in crops])),
		])


class GridSeq2Seq:
	def __init__(self, args, mode='train'):
		self.mode = mode
		self.args = args
		self.data = glob.glob(os.path.join(
			args.grid_root, 's*', '*_233feat.npz'))

		if mode == 'train':
			self.cur_data = [x for x in self.data if x.split(
				'/')[-2] not in ['s1', 's2', 's20', 's22']]
		elif mode == 'val':
			self.cur_data = [x for x in self.data if x.split(
				'/')[-2] in ['s1', 's2', 's20', 's22']]
			random.seed(1)
			random.shuffle(self.cur_data)
			self.cur_data = self.cur_data[:args.validation_batch*args.batch_size]
		else:
			self.cur_data = [x for x in self.data if x.split(
				'/')[-2] in ['s1', 's2', 's20', 's22']]

		self.character_list = ['^', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
							   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '*']
		self.character_dict = {e: idx for idx,
							   e in enumerate(self.character_list)}
		self.idx_dict = {idx: e for idx,
							   e in enumerate(self.character_list)}

	def __getitem__(self, index):
		item = self.cur_data[index]

		video = np.load(item)['data']
		align = self.align2idx(self.load_align(item.replace('_233feat.npz', '.align')))
		padded_align = self.align_pad(align)

		if self.mode != 'test':
			return video, np.array(padded_align)
		else:
			return video, np.array(padded_align), align, self.load_align(item.replace('_233feat.npz', '.align')), item

	def __len__(self):
		return len(self.cur_data)

	def load_align(self, name):
		with open(name, 'r') as f:
			lines = [line.strip().split(' ') for line in f.readlines()]
			txt = [line[2] for line in lines]
			txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
		return ' '.join(txt).upper()

	def align2idx(self, text):
		return [self.character_dict['*']]+[self.character_dict[x] for x in text]

	def align_pad(self, align):
		if len(align) == self.args.min_frame + 1:
			return align

		return align + [self.character_dict['^']] * (self.args.min_frame + 1 - len(align))


if __name__ == '__main__':
	opt = get_parser()
	crop_video(opt)
	crop_mouth(opt)
