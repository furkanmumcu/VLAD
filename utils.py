import torch
import torchvision.transforms as transform
import numpy as np
import math


def summary(probs, keywords):
	d = {}
	for i in range(len(keywords)):
		d[keywords[i]] = probs[0][i].item()
	sorted_dict = sorted(d.items(), key=lambda x: x[1], reverse=True)
	print('clip predictions: {}'.format(sorted_dict))


def kl_divergence(probs1, probs2):
	assert len(probs2) == len(probs1)

	summ = 0
	for k in range(len(probs1)):
		if probs2[k] == 0:
			probs2[k] = 1e-12
		try:
			summ = summ + (probs1[k] * math.log(probs1[k] / probs2[k]))
		except ValueError as error:
			print(error)

	return summ

def process_video(video):
	tr = transform.Resize((224, 224))
	video = np.transpose(video, (3, 0, 1, 2))
	video = torch.from_numpy(video)
	video = tr(video)
	video = torch.unsqueeze(video, 0)
	video = video.to(torch.float)
	video = video / 255
	video = video.to('cuda')
	return video


